import Foundation
import Capacitor
import CoreML
import Tokenizers     // huggingface/swift-transformers — AutoTokenizer, Tokenizer protocol

// MARK: - Event name constants

private let kEventTokenGenerated     = "onTokenGenerated"
private let kEventGenerationComplete = "onGenerationComplete"
private let kEventGenerationError    = "onGenerationError"

// MARK: - Generation configuration

struct GenerationConfig {
    var maxNewTokens: Int   = 200
    var temperature: Float  = 1.0
    var topK: Int           = 50
    var topP: Float         = 0.9
    var repetitionPenalty: Float = 1.1
    var eosTokenId: Int     = 2
}

// MARK: - CoreMLPlugin

// CAPBridgedPlugin replaces Plugin.m — pure Swift registration,
// no ObjC file needed, and compatible with SPM (which cannot mix languages).
@objc(CoreMLPlugin)
public class CoreMLPlugin: CAPPlugin, CAPBridgedPlugin {

    public let identifier = "CoreMLPlugin"
    public let jsName = "CoreMLPlugin"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "loadModel", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "generateText", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "unloadModel", returnType: CAPPluginReturnPromise),
    ]

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    private var model: MLModel?
    /// Holds an `MLState` on iOS 18+ for stateful (KV-cached) models.
    private var modelState: AnyObject?
    /// Production tokenizer loaded from bundled tokenizer.json via swift-transformers.
    private var tokenizer: (any Tokenizer)?
    private var isGenerating = false
    private var shouldCancelGeneration = false

    // -----------------------------------------------------------------------
    // MARK: loadModel
    // -----------------------------------------------------------------------

    @objc func loadModel(_ call: CAPPluginCall) {
        guard let modelName = call.getString("modelName"), !modelName.isEmpty else {
            call.reject("modelName is required and must not be empty.")
            return
        }
        let tokenizerFolderName = call.getString("tokenizerFolder") ?? modelName

        // Extract everything from `call` before entering the async boundary
        // so we never capture the non-Sendable CAPPluginCall across threads.
        call.keepAlive = true

        Task {
            // Evict previous model first
            self.unloadModelSync()

            do {
                // ── 1. Locate the CoreML model ────────────────────────────
                guard let modelURL = Bundle.main.url(forResource: modelName,
                                                     withExtension: "mlpackage")
                                  ?? Bundle.main.url(forResource: modelName,
                                                     withExtension: "mlmodelc") else {
                    call.reject("Model '\(modelName)' not found in bundle. " +
                                "Add the .mlpackage to your Xcode target membership.")
                    return
                }

                // ── 2. Locate the tokenizer folder ────────────────────────
                // Check subfolder first (blue folder reference), then bundle root
                // (yellow group / flat copy).
                let bundleURL = URL(fileURLWithPath: Bundle.main.bundlePath)
                let subfolderURL = bundleURL.appendingPathComponent(tokenizerFolderName)
                let tokenizerURL: URL
                if FileManager.default.fileExists(atPath:
                        subfolderURL.appendingPathComponent("tokenizer.json").path) {
                    tokenizerURL = subfolderURL
                } else if FileManager.default.fileExists(atPath:
                        bundleURL.appendingPathComponent("tokenizer.json").path) {
                    tokenizerURL = bundleURL
                } else {
                    call.reject(
                        "Tokenizer files not found in bundle (checked '\(tokenizerFolderName)/' " +
                        "subfolder and bundle root). Add tokenizer.json and " +
                        "tokenizer_config.json to your Xcode target."
                    )
                    return
                }

                // ── 3. Load the tokenizer (async, reads JSON from disk) ───
                let loadedTokenizer = try await AutoTokenizer.from(
                    modelFolder: tokenizerURL
                )

                // ── 4. Compile + load the CoreML model ────────────────────
                let mlConfig = MLModelConfiguration()
                mlConfig.computeUnits = .all

                let compiledURL: URL
                if modelURL.pathExtension == "mlpackage" {
                    compiledURL = try await MLModel.compileModel(at: modelURL)
                } else {
                    compiledURL = modelURL
                }

                // Use the async load() API — the synchronous MLModel(contentsOf:)
                // initializer crashes with EXC_BAD_ACCESS on large models due to
                // internal memory-mapping issues during ANE compilation.
                let loadedModel = try await MLModel.load(
                    contentsOf: compiledURL,
                    configuration: mlConfig
                )

                // ── 5. Store state ────────────────────────────────────────
                self.tokenizer = loadedTokenizer
                self.model     = loadedModel

                // Log model I/O so we can diagnose tensor mismatches
                let inputNames = loadedModel.modelDescription
                    .inputDescriptionsByName
                for (name, desc) in inputNames {
                    print("[CoreML] Input: \(name) → \(desc)")
                }
                let outputNames = loadedModel.modelDescription
                    .outputDescriptionsByName
                for (name, desc) in outputNames {
                    print("[CoreML] Output: \(name) → \(desc)")
                }

                // Always attempt to create MLState on iOS 18+.
                // If the model declares key_cache / value_cache inputs, CoreML
                // REQUIRES an MLState — calling prediction() without it throws
                // "The input feature for key_cache must be an MLState".
                if #available(iOS 18.0, *) {
                    self.modelState = loadedModel.makeState()
                    print("[CoreML] Created MLState for stateful inference (iOS 18+)")
                }

                let modelDesc = loadedModel.modelDescription
                    .metadata[MLModelMetadataKey.description] as? String

                call.resolve([
                    "success": true,
                    "modelDescription": modelDesc ?? "Model loaded successfully.",
                ])

            } catch {
                call.reject("loadModel failed: \(error.localizedDescription)", nil, error)
            }
        }
    }

    // -----------------------------------------------------------------------
    // MARK: generateText
    // -----------------------------------------------------------------------

    @objc func generateText(_ call: CAPPluginCall) {
        guard let model else {
            call.reject(CoreMLPluginError.modelNotLoaded.localizedDescription)
            return
        }
        guard let tokenizer else {
            call.reject(CoreMLPluginError.tokenizerNotLoaded.localizedDescription)
            return
        }
        guard !isGenerating else {
            call.reject("Generation already in progress. " +
                        "Await the current call or call unloadModel() to cancel.")
            return
        }
        guard let prompt = call.getString("prompt") else {
            call.reject("prompt parameter is required.")
            return
        }

        let eosTokenId = call.getInt("eosTokenId")
                      ?? tokenizer.eosTokenId
                      ?? 2

        let config = GenerationConfig(
            maxNewTokens:      call.getInt("maxNewTokens")              ?? 200,
            temperature:       Float(call.getFloat("temperature")       ?? 1.0),
            topK:              call.getInt("topK")                      ?? 50,
            topP:              Float(call.getFloat("topP")              ?? 0.9),
            repetitionPenalty: Float(call.getFloat("repetitionPenalty") ?? 1.1),
            eosTokenId:        eosTokenId
        )

        // Capture raw token IDs before crossing async boundary
        let rawTokenIds = call.getArray("tokenIds", Int.self)

        isGenerating = true
        shouldCancelGeneration = false
        call.keepAlive = true

        // Task inherits @MainActor but we immediately hop off via the
        // blocking MLModel.prediction calls which run on the cooperative pool.
        Task {
            defer { self.isGenerating = false }

            do {
                let promptIds: [Int]
                if let rawIds = rawTokenIds, !rawIds.isEmpty {
                    promptIds = rawIds
                } else {
                    promptIds = tokenizer.encode(text: prompt, addSpecialTokens: true)
                }

                print("[CoreML] Prompt token count: \(promptIds.count)")
                print("[CoreML] First 10 token IDs: \(Array(promptIds.prefix(10)))")
                print("[CoreML] EOS token ID: \(config.eosTokenId)")
                print("[CoreML] Tokenizer eosTokenId: \(String(describing: tokenizer.eosTokenId))")

                var generatedIds: [Int] = []
                var fullText = ""

                // Reset KV cache state for each new generation so the model
                // doesn't carry over context from the previous conversation.
                let useStatefulAPI: Bool
                if #available(iOS 18.0, *), self.model != nil {
                    self.modelState = model.makeState()
                    useStatefulAPI = self.modelState is MLState
                } else {
                    useStatefulAPI = false
                }
                print("[CoreML] Using stateful API: \(useStatefulAPI)")

                for step in 0..<config.maxNewTokens {
                    guard !self.shouldCancelGeneration else { break }

                    let inputIds: [Int]
                    let totalContextLen: Int  // full KV cache length for causal mask
                    if useStatefulAPI {
                        inputIds = (generatedIds.isEmpty ? promptIds : [generatedIds.last!])
                        totalContextLen = promptIds.count + generatedIds.count
                    } else {
                        inputIds = promptIds + generatedIds
                        totalContextLen = inputIds.count
                    }

                    let inputFeatures = try self.buildInput(
                        tokenIds: inputIds,
                        model: model,
                        isStateful: useStatefulAPI,
                        totalContextLen: totalContextLen
                    )

                    // Heavy inference — runs on cooperative thread pool
                    let output: MLFeatureProvider
                    if #available(iOS 18.0, *), let state = self.modelState as? MLState {
                        output = try model.prediction(from: inputFeatures, using: state)
                    } else {
                        output = try model.prediction(from: inputFeatures)
                    }

                    guard let nextId = try self.sampleNextToken(
                        from: output, config: config, generatedSoFar: generatedIds
                    ) else { break }

                    if nextId == config.eosTokenId { break }

                    generatedIds.append(nextId)

                    let tokenText = tokenizer.decode(
                        tokens: [nextId],
                        skipSpecialTokens: true
                    )
                    fullText += tokenText

                    self.notifyListeners(kEventTokenGenerated, data: [
                        "token":   tokenText,
                        "tokenId": nextId,
                        "index":   generatedIds.count - 1,
                        "text":    fullText,
                    ])
                }

                self.notifyListeners(kEventGenerationComplete, data: [
                    "text":       fullText,
                    "tokenCount": generatedIds.count,
                    "cancelled":  self.shouldCancelGeneration,
                ])
                call.resolve(["text": fullText, "tokenCount": generatedIds.count])

            } catch {
                self.notifyListeners(kEventGenerationError,
                                     data: ["error": error.localizedDescription])
                call.reject("generateText failed: \(error.localizedDescription)", nil, error)
            }
        }
    }

    // -----------------------------------------------------------------------
    // MARK: unloadModel
    // -----------------------------------------------------------------------

    @objc func unloadModel(_ call: CAPPluginCall) {
        shouldCancelGeneration = true
        unloadModelSync()
        call.resolve(["success": true])
    }

    // -----------------------------------------------------------------------
    // MARK: buildInput (private)
    // -----------------------------------------------------------------------

    /// Constructs the `MLFeatureProvider` for a single generation step.
    ///
    /// ⚠️ BRIDGING QUIRK #2 — Feature names are baked into the CoreML graph.
    /// Run this after loading to discover the exact names YOUR model uses:
    ///
    ///     let inputs  = model.modelDescription.inputDescriptionsByName.keys
    ///     let outputs = model.modelDescription.outputDescriptionsByName.keys
    ///     print(inputs, outputs)
    ///
    /// Common input names: "input_ids", "attention_mask", "position_ids"
    /// Common output names: "logits", "output_logits", "lm_logits"
    ///
    /// ⚠️ BRIDGING QUIRK #3 — MLMultiArray data type must match the model's
    /// declared constraint exactly.  Most exported LLMs expect Int32 for
    /// token IDs and Float32 for logits.  Passing Int64 where Int32 is
    /// expected causes a silent shape mismatch at prediction time.
    private func buildInput(tokenIds: [Int],
                            model: MLModel,
                            isStateful: Bool,
                            totalContextLen: Int) throws -> MLFeatureProvider {
        let seqLen = tokenIds.count
        let seqShape: [NSNumber] = [1, NSNumber(value: seqLen)]
        let inputDescs = model.modelDescription.inputDescriptionsByName

        // input_ids — [1, seqLen] Int32
        let inputIdsArray = try MLMultiArray(shape: seqShape, dataType: .int32)
        inputIdsArray.withUnsafeMutableBytes { raw, _ in
            let ptr = raw.bindMemory(to: Int32.self)
            for (i, id) in tokenIds.enumerated() { ptr[i] = Int32(id) }
        }

        var features: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
        ]

        // attention_mask — all 1s (only if the model declares it)
        if inputDescs["attention_mask"] != nil {
            let maskArray = try MLMultiArray(shape: seqShape, dataType: .int32)
            maskArray.withUnsafeMutableBytes { raw, _ in
                let ptr = raw.bindMemory(to: Int32.self)
                for i in 0..<seqLen { ptr[i] = 1 }
            }
            features["attention_mask"] = MLFeatureValue(multiArray: maskArray)
        }

        // causal_mask — attention mask for autoregressive generation.
        //
        // Shape: [1, 1, seqLen, totalContextLen]
        //   - seqLen = number of NEW tokens being processed this step
        //   - totalContextLen = full KV cache length (prompt + generated so far)
        //
        // For the prompt step (seqLen=N, totalCtx=N):
        //   Lower-triangular: row i attends to columns 0..i
        //
        // For generation steps (seqLen=1, totalCtx=N+step):
        //   Single row, all zeros: the new token attends to everything in the cache
        if let causalDesc = inputDescs["causal_mask"],
           let constraint = causalDesc.multiArrayConstraint {
            let rank = constraint.shape.count
            let dataType = constraint.dataType
            let keyDim = totalContextLen

            let maskShape: [NSNumber]
            if rank == 4 {
                maskShape = [1, 1, NSNumber(value: seqLen), NSNumber(value: keyDim)]
            } else if rank == 3 {
                maskShape = [1, NSNumber(value: seqLen), NSNumber(value: keyDim)]
            } else {
                maskShape = [NSNumber(value: seqLen), NSNumber(value: keyDim)]
            }

            let causalArray = try MLMultiArray(shape: maskShape, dataType: dataType)
            let totalElements = causalArray.count

            if dataType == .float16 {
                causalArray.withUnsafeMutableBytes { raw, _ in
                    let ptr = raw.bindMemory(to: UInt16.self)
                    let maskedVal = Float16(-1e4).bitPattern
                    let validVal  = Float16(0.0).bitPattern
                    // Init all to masked
                    for i in 0..<totalElements { ptr[i] = maskedVal }
                    // For each query row, unmask the causal window.
                    // startCol = position of the first token this row can see
                    //          = totalContextLen - seqLen (everything already in KV cache)
                    let cacheOffset = totalContextLen - seqLen
                    for row in 0..<seqLen {
                        // Attend to all cached tokens (0..<cacheOffset) + up to current position
                        let lastCol = cacheOffset + row
                        for col in 0...lastCol {
                            ptr[row * keyDim + col] = validVal
                        }
                    }
                }
            } else {
                causalArray.withUnsafeMutableBytes { raw, _ in
                    let ptr = raw.bindMemory(to: Float.self)
                    for i in 0..<totalElements { ptr[i] = -1e9 }
                    let cacheOffset = totalContextLen - seqLen
                    for row in 0..<seqLen {
                        let lastCol = cacheOffset + row
                        for col in 0...lastCol {
                            ptr[row * keyDim + col] = 0.0
                        }
                    }
                }
            }
            features["causal_mask"] = MLFeatureValue(multiArray: causalArray)
        }

        // position_ids — only add when the model explicitly declares this input.
        if inputDescs["position_ids"] != nil {
            let posArray = try MLMultiArray(shape: seqShape, dataType: .int32)
            let posOffset = isStateful ? (totalContextLen - seqLen) : 0
            posArray.withUnsafeMutableBytes { raw, _ in
                let ptr = raw.bindMemory(to: Int32.self)
                for i in 0..<seqLen { ptr[i] = Int32(posOffset + i) }
            }
            features["position_ids"] = MLFeatureValue(multiArray: posArray)
        }

        #if DEBUG
        let provided = Set(features.keys)
        let required = Set(inputDescs.keys)
        let missing = required.subtracting(provided)
            .filter { !$0.contains("cache") && !$0.contains("state") }
        if !missing.isEmpty {
            print("[CoreML] ⚠️ Missing input features: \(missing)")
        }
        #endif

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    // -----------------------------------------------------------------------
    // MARK: sampleNextToken (private)
    // -----------------------------------------------------------------------

    /// Extracts the next-token prediction from the model's logit output.
    ///
    /// ⚠️ BRIDGING QUIRK #5 — Logit tensor layout:
    ///   Non-stateful: shape [1, seqLen, vocabSize] — use the LAST position.
    ///   Stateful:     shape [1, 1,      vocabSize] — position 0 is always correct.
    ///
    /// ⚠️ BRIDGING QUIRK #6 — Float16 vs Float32:
    /// Some models (especially ANE-optimised ones) output Float16 logits.
    /// MLMultiArray.dataType will be `.float16`; you must read the raw bytes
    /// as UInt16 and convert via `Float(Float16(bitPattern: uint16Value))`.
    /// The code below handles both Float32 and Float16 transparently.
    private func sampleNextToken(from output: MLFeatureProvider,
                                  config: GenerationConfig,
                                  generatedSoFar: [Int]) throws -> Int? {
        // Try the most common logits output feature names.
        let candidateNames = ["logits", "output_logits", "lm_logits", "output"]
        guard let featureName = candidateNames.first(where: {
                  output.featureValue(for: $0) != nil
              }),
              let logitsArray = output.featureValue(for: featureName)?.multiArrayValue
        else { throw CoreMLPluginError.missingLogits }

        let shape = logitsArray.shape.map { $0.intValue }
        guard shape.count >= 2 else { throw CoreMLPluginError.invalidLogitsShape }

        let vocabSize = shape[shape.count - 1]
        // Last-token offset in the flat buffer
        let lastPos   = shape.count == 3 ? (shape[1] - 1) * vocabSize : 0

        // Fast bulk read — avoids per-element NSNumber boxing overhead
        var logits = [Float](repeating: 0, count: vocabSize)
        logitsArray.withUnsafeBytes { rawPtr in
            switch logitsArray.dataType {
            case .float32:
                let src = rawPtr.bindMemory(to: Float.self)
                for i in 0..<vocabSize { logits[i] = src[lastPos + i] }
            case .float16:
                let src = rawPtr.bindMemory(to: UInt16.self)
                for i in 0..<vocabSize {
                    logits[i] = Float(Float16(bitPattern: src[lastPos + i]))
                }
            default:  // fallback for other types
                let src = rawPtr.bindMemory(to: Float.self)
                for i in 0..<vocabSize { logits[i] = src[lastPos + i] }
            }
        }

        // Repetition penalty — scale down logits for tokens already generated
        if config.repetitionPenalty != 1.0 {
            for id in generatedSoFar where id < vocabSize {
                logits[id] = logits[id] < 0
                    ? logits[id] * config.repetitionPenalty
                    : logits[id] / config.repetitionPenalty
            }
        }

        // Temperature scaling
        if config.temperature > 0.01 {
            logits = logits.map { $0 / config.temperature }
            return topKTopPSample(logits: logits, topK: config.topK, topP: config.topP)
        } else {
            // Greedy argmax
            return logits.indices.max(by: { logits[$0] < logits[$1] })
        }
    }

    // -----------------------------------------------------------------------
    // MARK: topKTopPSample (private)
    // -----------------------------------------------------------------------

    private func topKTopPSample(logits: [Float], topK: Int, topP: Float) -> Int {
        let sortedIndices = logits.indices.sorted { logits[$0] > logits[$1] }
        let kIndices = Array(sortedIndices.prefix(min(topK, sortedIndices.count)))

        // Numerically stable softmax over top-K candidates
        let maxLogit = kIndices.map { logits[$0] }.max() ?? 0
        var probs = kIndices.map { expf(logits[$0] - maxLogit) }
        let sum = probs.reduce(0, +)
        probs = probs.map { $0 / sum }

        // Nucleus (top-P) filtering
        var cumulative: Float = 0
        var nucleusIdx: [Int]   = []
        var nucleusProbs: [Float] = []
        for (i, idx) in kIndices.enumerated() {
            cumulative += probs[i]
            nucleusIdx.append(idx)
            nucleusProbs.append(probs[i])
            if cumulative >= topP { break }
        }

        // Re-normalise and sample
        let nucleusSum = nucleusProbs.reduce(0, +)
        let normalized = nucleusProbs.map { $0 / nucleusSum }

        var r = Float.random(in: 0..<1)
        for (i, p) in normalized.enumerated() {
            r -= p
            if r <= 0 { return nucleusIdx[i] }
        }
        return nucleusIdx.last ?? 0
    }

    // -----------------------------------------------------------------------
    // MARK: unloadModelSync (private)
    // -----------------------------------------------------------------------

    /// Releases all model + tokenizer memory synchronously.
    private func unloadModelSync() {
        model      = nil
        modelState = nil   // releases the KV-cache memory block
        tokenizer  = nil   // releases the BPE vocabulary from memory
    }
}

// MARK: - Plugin errors

enum CoreMLPluginError: LocalizedError {
    case missingLogits
    case invalidLogitsShape
    case modelNotLoaded
    case tokenizerNotLoaded

    var errorDescription: String? {
        switch self {
        case .missingLogits:
            return "No 'logits' output found.  Inspect " +
                   "model.modelDescription.outputDescriptionsByName.keys and " +
                   "update the logitsFeatureName search list in sampleNextToken()."
        case .invalidLogitsShape:
            return "Logits tensor rank < 2.  Expected [1, seqLen, vocabSize] or [1, vocabSize]."
        case .modelNotLoaded:
            return "No model is loaded.  Call loadModel() first."
        case .tokenizerNotLoaded:
            return "Tokenizer not loaded.  Ensure tokenizer.json and " +
                   "tokenizer_config.json are bundled and loadModel() succeeded."
        }
    }
}
