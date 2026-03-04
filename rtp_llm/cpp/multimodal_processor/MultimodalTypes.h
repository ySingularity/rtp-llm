#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputClass.h"

namespace rtp_llm {
struct MultimodalOutput {
    std::vector<torch::Tensor>                mm_features         = {};
    std::optional<std::vector<torch::Tensor>> mm_position_ids     = std::nullopt;
    std::optional<std::vector<torch::Tensor>> mm_deepstack_embeds = std::nullopt;
};

class MultimodalFeature {
public:
    std::vector<torch::Tensor>   features;
    std::vector<MultimodalInput> inputs;
    rtp_llm::BufferPtr           text_tokens_mask;  // text part for 1 and multimodal part for 0
    rtp_llm::BufferPtr           locs;              // multimodal input locations
    rtp_llm::BufferPtr           expanded_ids;
    MultimodalFeature() {}
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "MultimodalFeature {"
                     << "features: " << features.size() << ", inputs: " << inputs.size()
                     << ", text_tokens_mask: " << text_tokens_mask->debugStringWithData<int32_t>()
                     << ", locs: " << locs->debugStringWithData<int32_t>()
                     << ", expanded_ids: " << expanded_ids->debugStringWithData<int32_t>() << "}";
        return debug_string.str();
    }
};

}  // namespace rtp_llm
