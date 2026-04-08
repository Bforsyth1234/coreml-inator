import type { PluginListenerHandle } from '@capacitor/core';

// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------

export interface LoadModelOptions {
  /**
   * The bare name of the .mlpackage (or pre-compiled .mlmodelc) bundled inside
   * your Xcode target.  Do NOT include the file extension.
   * e.g. "Llama3_2_1B_Instruct_4bit"
   */
  modelName: string;

  /**
   * Name of the **folder** (added as a blue folder reference in Xcode) that
   * contains `tokenizer.json` and `tokenizer_config.json`.
   *
   * When omitted the plugin looks for a folder whose name matches `modelName`.
   * That is, if modelName = "Llama3_2_1B" the plugin expects a bundle folder
   * called "Llama3_2_1B/" holding the two tokenizer files.
   *
   * Obtain these files from the same HuggingFace repo as your model weights:
   *   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
   *     tokenizer.json tokenizer_config.json \
   *     --local-dir ios/App/App/Llama3_2_1B
   */
  tokenizerFolder?: string;
}

export interface LoadModelResult {
  success: boolean;
  /** Free-text description embedded in the CoreML model metadata, if present. */
  modelDescription?: string;
}

// ---------------------------------------------------------------------------
// generateText
// ---------------------------------------------------------------------------

export interface GenerateTextOptions {
  /**
   * Natural-language prompt that will be encoded by the Swift tokenizer.
   * If you perform tokenization yourself in JavaScript, pass `tokenIds`
   * instead and leave this empty.
   */
  prompt: string;

  /**
   * Pre-encoded token IDs.  When provided the Swift side skips its own
   * tokenizer and feeds these IDs directly to the model.  Useful when you
   * bundle a WASM/JS tokenizer (e.g. @huggingface/transformers) in the web
   * layer for accuracy.
   */
  tokenIds?: number[];

  /** Maximum number of *new* tokens to generate (default: 200). */
  maxNewTokens?: number;

  /**
   * Sampling temperature.  1.0 = unchanged distribution.
   * Lower → more deterministic.  0.0 → greedy argmax.  (default: 1.0)
   */
  temperature?: number;

  /** Top-K candidates kept before nucleus filtering (default: 50). */
  topK?: number;

  /** Nucleus (Top-P) probability mass threshold (default: 0.9). */
  topP?: number;

  /** Penalises tokens that already appear in the output (default: 1.1). */
  repetitionPenalty?: number;

  /**
   * Token ID that signals end-of-sequence.  Generation stops when this
   * token is produced.  Defaults to the tokenizer's own `eosTokenId`.
   */
  eosTokenId?: number;
}

export interface GenerateTextResult {
  /** Full generated text (detokenised). */
  text: string;
  /** Number of tokens that were generated before EOS / maxNewTokens. */
  tokenCount: number;
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

export interface TokenEvent {
  /** Detokenised string for this single token (may be a sub-word fragment). */
  token: string;
  /** Raw vocabulary index returned by the model. */
  tokenId: number;
  /** Zero-based position in the generated sequence. */
  index: number;
  /** Full accumulated text from position 0 through this token. */
  text: string;
}

export interface GenerationCompleteEvent {
  text: string;
  tokenCount: number;
  /** True when the caller triggered cancellation via unloadModel(). */
  cancelled: boolean;
}

export interface GenerationErrorEvent {
  error: string;
}

// ---------------------------------------------------------------------------
// Plugin interface
// ---------------------------------------------------------------------------

export interface CoreMLPlugin {
  /**
   * Loads and compiles the named CoreML model with computeUnits = .all so
   * that inference is routed to the Apple Neural Engine when available.
   */
  loadModel(options: LoadModelOptions): Promise<LoadModelResult>;

  /**
   * Begins text generation on a background thread.  Individual tokens are
   * streamed back via the `onTokenGenerated` event as they are produced.
   * The returned promise resolves only after generation is complete.
   */
  generateText(options: GenerateTextOptions): Promise<GenerateTextResult>;

  /**
   * Sets a cancellation flag, waits for the in-flight generation loop to
   * exit, then releases the MLModel from memory.  Call this when the
   * feature is no longer needed to avoid the iOS watchdog killing the app.
   */
  unloadModel(): Promise<{ success: boolean }>;

  addListener(
    eventName: 'onTokenGenerated',
    listenerFunc: (event: TokenEvent) => void,
  ): Promise<PluginListenerHandle> & PluginListenerHandle;

  addListener(
    eventName: 'onGenerationComplete',
    listenerFunc: (event: GenerationCompleteEvent) => void,
  ): Promise<PluginListenerHandle> & PluginListenerHandle;

  addListener(
    eventName: 'onGenerationError',
    listenerFunc: (event: GenerationErrorEvent) => void,
  ): Promise<PluginListenerHandle> & PluginListenerHandle;

  removeAllListeners(): Promise<void>;
}
