#import "TorchModule.h"
#import <Foundation/Foundation.h>

// Include C++ APIs directly in Objective-C++
#pragma clang diagnostic push
#if __has_warning("-Wdeprecated-declarations")
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if __has_warning("-Wcomma")
#pragma clang diagnostic ignored "-Wcomma"
#endif
#if __has_warning("-Wshorten-64-to-32")
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#endif
#include <torch/script.h>
#include <c10/core/InferenceMode.h>
#pragma clang diagnostic pop

@interface TorchModule () {
    std::shared_ptr<torch::jit::Module> _module;
}
@end

@implementation TorchModule

- (nullable instancetype)initWithFile:(NSString *)filePath {
    self = [super init];
    if (self) {
        try {
            std::string path([filePath UTF8String]);
            // Preflight: check file exists and log size
            @autoreleasepool {
                BOOL isDir = NO;
                BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:filePath isDirectory:&isDir];
                if (!exists || isDir) {
                    NSLog(@"Torch load error: file not found at %@", filePath);
                    return nil;
                }
                NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:filePath error:nil];
                unsigned long long size = [attrs fileSize];
                NSLog(@"Loading TorchScript model: %@ (size=%llu bytes)", filePath, size);
            }
            torch::jit::Module m = torch::jit::load(path);
            m.eval();
            _module = std::make_shared<torch::jit::Module>(std::move(m));
        } catch (const c10::Error &e) {
            NSLog(@"Torch load error: %s", e.what());
            return nil;
        } catch (const std::exception &e) {
            NSLog(@"Torch std::exception: %s", e.what());
            return nil;
        } catch (...) {
            NSLog(@"Torch load unknown error");
            return nil;
        }
    }
    return self;
}

- (nullable NSArray<NSNumber *> *)embedAudioPCM:(NSArray<NSNumber *> *)pcm16kMono {
    if (!_module) { return nil; }
    // Convert NSArray<NSNumber(float)> to raw samples
    std::vector<float> wav;
    wav.reserve(pcm16kMono.count);
    for (NSNumber *n in pcm16kMono) {
        wav.push_back([n floatValue]);
    }
    // Attempt 1: feed 1-D vector [T] (matches traced wrapper example input)
    torch::Tensor out;
    {
        torch::Tensor input1d = torch::from_blob(wav.data(), {(long long)wav.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        NSLog(@"Torch forward attempt A (1D): input shape = (%lld)", (long long)wav.size());
        std::vector<torch::jit::IValue> inputsA;
        inputsA.emplace_back(input1d);
        try {
            c10::InferenceMode guard(true);
            auto iv = _module->forward(inputsA);
            if (iv.isTensor()) { out = iv.toTensor(); }
            else if (iv.isTuple()) {
                auto tup = iv.toTuple();
                for (const auto &el : tup->elements()) { if (el.isTensor()) { out = el.toTensor(); break; } }
            } else if (iv.isGenericDict()) {
                auto dict = iv.toGenericDict();
                for (const auto &kv : dict) { if (kv.value().isTensor()) { out = kv.value().toTensor(); break; } }
            }
        } catch (const c10::Error &e) {
            NSLog(@"Torch forward error (1D): %s", e.what());
        } catch (const std::exception &e) {
            NSLog(@"Torch forward std::exception (1D): %s", e.what());
        } catch (...) {
            NSLog(@"Torch forward unknown error (1D)");
        }
    }
    // Attempt 2: if needed, feed 2-D [1, T]
    if (!out.defined()) {
        torch::Tensor input2d = torch::from_blob(wav.data(), {(long long)1, (long long)wav.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        NSLog(@"Torch forward attempt B (2D): input shape = (1, %lld)", (long long)wav.size());
        std::vector<torch::jit::IValue> inputsB;
        inputsB.emplace_back(input2d);
        try {
            c10::InferenceMode guard(true);
            auto iv = _module->forward(inputsB);
            if (iv.isTensor()) { out = iv.toTensor(); }
            else if (iv.isTuple()) {
                auto tup = iv.toTuple();
                for (const auto &el : tup->elements()) { if (el.isTensor()) { out = el.toTensor(); break; } }
            } else if (iv.isGenericDict()) {
                auto dict = iv.toGenericDict();
                for (const auto &kv : dict) { if (kv.value().isTensor()) { out = kv.value().toTensor(); break; } }
            }
        } catch (const c10::Error &e) {
            NSLog(@"Torch forward error (2D): %s", e.what());
        } catch (const std::exception &e) {
            NSLog(@"Torch forward std::exception (2D): %s", e.what());
        } catch (...) {
            NSLog(@"Torch forward unknown error (2D)");
        }
    }

    if (!out.defined()) { return nil; }
    if (out.dim() == 2 && out.size(0) == 1) {
        out = out.squeeze(0);
    }
    // Log output sizes
    try {
        auto sizes = out.sizes();
        std::string sz = "(";
        for (size_t i = 0; i < sizes.size(); ++i) {
            sz += std::to_string(sizes[i]);
            if (i + 1 < sizes.size()) sz += ", ";
        }
        sz += ")";
        NSLog(@"Torch forward: output shape = %s", sz.c_str());
    } catch (...) {}
    out = out.contiguous();
    std::vector<float> emb(out.numel());
    std::memcpy(emb.data(), out.data_ptr<float>(), emb.size() * sizeof(float));

    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:emb.size()];
    for (float v : emb) { [result addObject:@(v)]; }
    return result;
}

// End-to-end scorer: wav mono at given sample rate -> normalized embedding
- (nullable NSArray<NSNumber *> *)scoreEmbedFromPCM:(NSArray<NSNumber *> *)pcmMono
                                    sampleRateHz:(double)sr {
    if (!_module) { return nil; }
    std::vector<float> wav;
    wav.reserve(pcmMono.count);
    for (NSNumber *n in pcmMono) {
        wav.push_back([n floatValue]);
    }
    torch::Tensor out;
    try {
        c10::InferenceMode guard(true);
        torch::Tensor input1d = torch::from_blob(wav.data(), {(long long)wav.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input1d);
        inputs.emplace_back((int64_t)sr);
        auto iv = _module->forward(inputs);
        if (iv.isTensor()) {
            out = iv.toTensor();
        } else if (iv.isTuple()) {
            auto tup = iv.toTuple();
            for (const auto &el : tup->elements()) { if (el.isTensor()) { out = el.toTensor(); break; } }
        }
    } catch (const c10::Error &e) {
        NSLog(@"Torch scorer error: %s", e.what());
        return nil;
    } catch (const std::exception &e) {
        NSLog(@"Torch scorer std::exception: %s", e.what());
        return nil;
    } catch (...) {
        NSLog(@"Torch scorer unknown error");
        return nil;
    }
    if (!out.defined()) { return nil; }
    out = out.contiguous();
    std::vector<float> emb(out.numel());
    std::memcpy(emb.data(), out.data_ptr<float>(), emb.size() * sizeof(float));
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:emb.size()];
    for (float v : emb) { [result addObject:@(v)]; }
    return result;
}

// NSNumber convenience
- (nullable NSArray<NSNumber *> *)scoreEmbedFromPCM:(NSArray<NSNumber *> *)pcmMono
                            sampleRateNumber:(NSNumber *)srNumber {
    return [self scoreEmbedFromPCM:pcmMono sampleRateHz:[srNumber doubleValue]];
}

@end


