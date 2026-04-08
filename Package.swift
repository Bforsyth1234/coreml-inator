// swift-tools-version: 5.9
import PackageDescription

// This Package.swift lets Capacitor's `cap sync` automatically include
// coreml-inator in the host app's CapApp-SPM umbrella package.
// It also declares the swift-transformers dependency so Xcode resolves it
// without any manual configuration.
let package = Package(
    name: "CoremlInator",
    platforms: [.iOS(.v16)],
    products: [
        .library(
            name: "CoremlInator",
            targets: ["CoreMLPlugin"]
        )
    ],
    dependencies: [
        // Capacitor's Swift PM bridge — 8.0.0..<9.0.0 matches all Capacitor 8.x hosts.
        .package(url: "https://github.com/ionic-team/capacitor-swift-pm.git",
                 from: "8.0.0"),
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
                .product(name: "Tokenizers",   package: "swift-transformers"),
            ],
            // Plugin.swift lives here (pure Swift, no ObjC).
            path: "ios/Plugin"
        )
    ]
)
