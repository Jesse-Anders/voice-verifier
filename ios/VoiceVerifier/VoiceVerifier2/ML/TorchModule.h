#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

- (nullable instancetype)initWithFile:(NSString *)filePath;
- (nullable NSArray<NSNumber *> *)embedAudioPCM:(NSArray<NSNumber *> *)pcm16kMono;
// End-to-end scorer: mono float32 at sampleRateHz -> normalized embedding
- (nullable NSArray<NSNumber *> *)scoreEmbedFromPCM:(NSArray<NSNumber *> *)pcmMono
                                    sampleRateHz:(double)sr;
// Convenience wrapper accepting NSNumber for sample rate (helps dynamic invocation from Swift)
- (nullable NSArray<NSNumber *> *)scoreEmbedFromPCM:(NSArray<NSNumber *> *)pcmMono
                            sampleRateNumber:(NSNumber *)srNumber;

@end

NS_ASSUME_NONNULL_END


