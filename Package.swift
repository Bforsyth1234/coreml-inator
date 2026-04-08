// swift-tools-version: 5.9
import PackageDescription

// This Package.swift lets Capacitor's `cap sync` automatically include
// coreml-inator in the host app's CapApp-SPM umbrella package.
// It also declares the swift-transformers dependency so Xcode resolves it
// without any manual configuration.
let package = Package(
    name: "CoreMLInator",
    platforms: [.iOS(.v16)],
    products: [
        .library(
            name: "CoreMLInator",
            targets: ["CoreMLPlugin"]
        )
    ],
    dependencies: [
        // Capacitor's Swift PM bridge — version is resolved by the host app.
        .package(url: "https://github.com/ionic-team/capacitor-swift-pm.git",
                 branch: "main"),
        // HuggingFace tokenizer library (BPE / SentencePiece).
        .package(url: "https://github.com/huggingface/swift-transformers",
                 from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "CoreMLPlugin",
            dependencies: [
                .product(name: "Capacitor",    package: "capacitor-swift-pm"),
                .product(name: "Cordova",      package: "capacitor-swift-pm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            // Both Plugin.swift and Plugin.m live here.
            path: "ios/Plugin",
            // Exposes the directory as a public headers path so the ObjC
            // CAP_PLUGIN macro can resolve Capacitor's umbrella header.
            publicHeadersPath: "."
        )
    ]
)
