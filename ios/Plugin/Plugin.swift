import Foundation
import Capacitor
import CoreML
import Transformers   // huggingface/swift-transformers — AutoTokenizer, Tokenizer protocol

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

@objc(CoreMLPlugin)
public class CoreMLPlugin: CAPPlugin {

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
        // tokenizerFolder defaults to the same name as the model.
        let tokenizerFolderName = call.getString("tokenizerFolder") ?? modelName

        // Task.detached runs on the Swift cooperative thread pool — never blocks
        // the main thread, and avoids the DispatchQueue/async-await impedance
        // mismatch since AutoTokenizer.from(modelFolder:) is async throws.
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await self.unloadModelAsync()

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
                // The folder must be added to Xcode as a blue "folder reference"
                // (not a group) so its contents are copied flat into the bundle.
                let bundlePath = Bundle.main.bundlePath
                let tokenizerURL = URL(fileURLWithPath: bundlePath)
                    .appendingPathComponent(tokenizerFolderName)

                guard FileManager.default.fileExists(atPath:
                        tokenizerURL.appendingPathComponent("tokenizer.json").path) else {
                    call.reject(
                        "Tokenizer files not found at bundle path '\(tokenizerFolderName)/'. " +
                        "Add tokenizer.json and tokenizer_config.json as a blue folder " +
                        "reference in Xcode and verify the folder name matches 'tokenizerFolder'."
                    )
                    return
                }

                // ── 3. Load the tokenizer (async, reads JSON from disk) ───
                // AutoTokenizer.from(modelFolder:) parses tokenizer_config.json
                // to determine the correct tokenizer class (BPE, SentencePiece,
                // WordPiece, etc.) and loads its vocabulary — all offline.
                let loadedTokenizer = try await AutoTokenizer.from(
                    modelFolder: tokenizerURL
                )

                // ── 4. Compile + load the CoreML model (sync, blocking) ───
                // .all routes inference to ANE → GPU → CPU in priority order.
                // Never use .cpuOnly for LLMs — it is 10-50× slower.
                let mlConfig = MLModelConfiguration()
                mlConfig.computeUnits = .all

                let compiledURL: URL
                if modelURL.pathExtension == "mlpackage" {
                    compiledURL = try MLModel.compileModel(at: modelURL)
                } else {
                    compiledURL = modelURL
                }
                let loadedModel = try MLModel(contentsOf: compiledURL,
                                              configuration: mlConfig)

                // ── 5. Capture state (actor-isolated to avoid data races) ─
                await MainActor.run {
                    self.tokenizer  = loadedTokenizer
                    self.model      = loadedModel
                }

                // iOS 18+ stateful API: MLState persists the KV cache so we
                // feed ONE new token per step rather than the full context window.
                if #available(iOS 18.0, *) {
                    let state = loadedModel.makeState()
                    await MainActor.run { self.modelState = state }
                }

                let desc = loadedModel.modelDescription
                    .metadata[MLModelMetadataKey.description] as? String

                call.resolve([
                    "success": true,
                    "modelDescription": desc ?? "Model loaded successfully.",
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

        // EOS: explicit JS override → tokenizer's own eosTokenId → fallback 2
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

        isGenerating = true
        shouldCancelGeneration = false

        // ⚡ THREADING — ALL CoreML matrix math runs here, never on main.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            defer { DispatchQueue.main.async { self.isGenerating = false } }

            do {
                // ── Encode prompt ─────────────────────────────────────────
                // Bypass the Swift tokenizer when JS passes raw token IDs
                // (e.g. if you are using @huggingface/transformers in a WASM
                // worker and want bit-for-bit identical tokenization to Python).
                let promptIds: [Int]
                if let rawIds = call.getArray("tokenIds", Int.self), !rawIds.isEmpty {
                    promptIds = rawIds
                } else {
                    // encode(text:addSpecialTokens:) handles BOS prepending,
                    // SentencePiece normalisation, BPE merges — all from the
                    // bundled tokenizer.json.
                    promptIds = tokenizer.encode(text: prompt, addSpecialTokens: true)
                }

                var generatedIds: [Int] = []
                var fullText = ""

                let useStatefulAPI: Bool
                if #available(iOS 18.0, *), self.modelState is MLState {
                    useStatefulAPI = true
                } else {
                    useStatefulAPI = false
                }

                for _ in 0..<config.maxNewTokens {
                    guard !self.shouldCancelGeneration else { break }

                    let inputIds = useStatefulAPI
                        ? (generatedIds.isEmpty ? promptIds : [generatedIds.last!])
                        : promptIds + generatedIds

                    let inputFeatures = try self.buildInput(tokenIds: inputIds,
                                                            model: model,
                                                            isStateful: useStatefulAPI)

                    // --- THE INFERENCE CALL ---
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

                    // ── Decode single token ───────────────────────────────
                    // decode(tokens:skipSpecialTokens:) converts one token ID
                    // to its UTF-8 string fragment.  skipSpecialTokens: true
                    // suppresses <pad>, <eos>, etc. from appearing in the UI.
                    let tokenText = tokenizer.decode(
                        tokens: [nextId],
                        skipSpecialTokens: true
                    )
                    fullText += tokenText

                    // 🔔 Stream to JavaScript — Capacitor dispatches to main thread.
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
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await self.unloadModelAsync()
            call.resolve(["success": true])
        }
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
                            isStateful: Bool) throws -> MLFeatureProvider {
        let seqLen = tokenIds.count
        let seqShape: [NSNumber] = [1, NSNumber(value: seqLen)]

        // input_ids — [1, seqLen] Int32
        let inputIdsArray = try MLMultiArray(shape: seqShape, dataType: .int32)
        // ⚠️ BRIDGING QUIRK #4 — Use withUnsafeMutableBytes for bulk writes.
        // The subscript operator re-validates bounds on every access and is
        // ~100× slower for large tensors.
        inputIdsArray.withUnsafeMutableBytes { raw in
            let ptr = raw.bindMemory(to: Int32.self)
            for (i, id) in tokenIds.enumerated() { ptr[i] = Int32(id) }
        }

        // attention_mask — all 1s, same shape
        let maskArray = try MLMultiArray(shape: seqShape, dataType: .int32)
        maskArray.withUnsafeMutableBytes { raw in
            let ptr = raw.bindMemory(to: Int32.self)
            for i in 0..<seqLen { ptr[i] = 1 }
        }

        var features: [String: MLFeatureValue] = [
            "input_ids":      MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ]

        // position_ids — only add when the model explicitly declares this input.
        // Stateful models usually manage position tracking internally via MLState.
        if !isStateful,
           model.modelDescription.inputDescriptionsByName["position_ids"] != nil {
            let posArray = try MLMultiArray(shape: seqShape, dataType: .int32)
            posArray.withUnsafeMutableBytes { raw in
                let ptr = raw.bindMemory(to: Int32.self)
                for i in 0..<seqLen { ptr[i] = Int32(i) }
            }
            features["position_ids"] = MLFeatureValue(multiArray: posArray)
        }

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
    // MARK: unloadModelAsync (private)
    // -----------------------------------------------------------------------

    /// Safely releases all model + tokenizer memory.
    /// Called from Task.detached so @MainActor.run is available to isolate
    /// the property writes from the generation loop's DispatchQueue thread.
    @MainActor
    private func unloadModelAsync() {
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
