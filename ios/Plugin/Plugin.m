#import <Foundation/Foundation.h>
#import <Capacitor/Capacitor.h>

/**
 * CoreMLPlugin — Objective-C bridge registration.
 *
 * Each CAP_PLUGIN_METHOD entry must exactly match the @objc method name
 * declared in Plugin.swift.  CAPPluginReturnPromise means the JS caller
 * receives a Promise (resolve / reject).  CAPPluginReturnNone would be used
 * for fire-and-forget calls, but we always want Promise-based error handling.
 */
CAP_PLUGIN(CoreMLPlugin, "CoreMLPlugin",
    CAP_PLUGIN_METHOD(loadModel,    CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(generateText, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(unloadModel,  CAPPluginReturnPromise);
)
