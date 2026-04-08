import { registerPlugin } from '@capacitor/core';

import type { CoreMLPlugin } from './definitions';

/**
 * Capacitor bridge to the native CoreML inference engine.
 *
 * Usage
 * -----
 * ```ts
 * import { CoreML } from 'coreml-inator';
 *
 * // 1. Load the model (once, expensive – do this at app start or lazily)
 * await CoreML.loadModel({ modelName: 'Llama3_2_1B_Instruct_4bit' });
 *
 * // 2. Subscribe to the token stream BEFORE calling generateText
 * const handle = await CoreML.addListener('onTokenGenerated', ({ token, text }) => {
 *   console.log('new token:', token);
 *   myTextView.innerText = text;
 * });
 *
 * // 3. Start generation (resolves when complete)
 * const { tokenCount } = await CoreML.generateText({
 *   prompt: 'Explain neural engines in one paragraph.',
 *   maxNewTokens: 200,
 *   temperature: 0.7,
 * });
 *
 * // 4. Clean up listener and model when done
 * await handle.remove();
 * await CoreML.unloadModel();
 * ```
 */
// CoreML is Apple-on-device only.  There is intentionally no web fallback —
// calls made from a non-iOS context will reject with "not implemented".
const CoreML = registerPlugin<CoreMLPlugin>('CoreMLPlugin');

export * from './definitions';
export { CoreML };
