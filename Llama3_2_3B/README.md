---
language:
- en
- de
- fr
- it
- pt
- hi
- es
- th
library_name: coreml
license: llama3.2
pipeline_tag: text-generation
tags:
- facebook
- meta
- llama
- llama-3
- CoreMLPipelines
---

# coreml-Llama-3.2-3B-Instruct-4bit

This model was converted from [meta-llama/Llama-3.2-3B-Instruct](https://hf.co/meta-llama/Llama-3.2-3B-Instruct) to CoreML using [coremlpipelinestools](https://github.com/finnvoor/CoreMLPipelines/tree/main/coremlpipelinestools).

### Use with [CoreMLPipelines](https://github.com/finnvoor/CoreMLPipelines)

```swift
import CoreMLPipelines

let pipeline = try await TextGenerationPipeline(
    modelName: "finnvoorhees/coreml-Llama-3.2-3B-Instruct-4bit"
)
let stream = pipeline(
    messages: [[
        "role": "user",
        "content": "Write a poem about Ireland"
    ]]
)
for try await text in stream {
    print(text, terminator: "")
    fflush(stdout)
}
print("")
```
    