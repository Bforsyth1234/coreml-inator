/**
 * rollup.config.mjs
 *
 * Reads the ESM output that tsc already compiled into dist/esm/ and produces
 * two additional bundle formats required by the Capacitor plugin contract:
 *
 *   dist/plugin.cjs.js  — CommonJS   (consumed by `require()`, older toolchains)
 *   dist/plugin.js      — IIFE       (loaded via <script> tag / unpkg CDN)
 *
 * @capacitor/core is declared as external so it is never bundled — the host
 * app always supplies it, either from node_modules (CJS/ESM) or from the
 * global `capacitorExports` object that Capacitor injects at runtime (IIFE).
 *
 * No TypeScript plugin is needed here because tsc handles compilation in the
 * first half of the build script (`npm run clean && tsc && rollup …`).
 */

export default [
  // ── CommonJS bundle ──────────────────────────────────────────────────────
  // Used by Jest, older webpack configs, and any toolchain that calls require().
  {
    input: 'dist/esm/index.js',
    output: {
      file: 'dist/plugin.cjs.js',
      format: 'cjs',
      sourcemap: true,
      // inlineDynamicImports: required when the ESM source contains
      // dynamic import() expressions (e.g. lazy-loaded platform code).
      inlineDynamicImports: true,
    },
    external: ['@capacitor/core'],
  },

  // ── IIFE bundle ───────────────────────────────────────────────────────────
  // Served via the `unpkg` field in package.json.  The global name must be a
  // valid JavaScript identifier; Capacitor's runtime will find the plugin by
  // scanning window['capacitorCoreMLinator'].
  {
    input: 'dist/esm/index.js',
    output: {
      file: 'dist/plugin.js',
      format: 'iife',
      name: 'capacitorCoreMLinator',
      // Map the @capacitor/core peer to the global that Capacitor injects.
      globals: {
        '@capacitor/core': 'capacitorExports',
      },
      sourcemap: true,
      inlineDynamicImports: true,
    },
    external: ['@capacitor/core'],
  },
];
