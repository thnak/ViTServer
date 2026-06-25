#pragma once

#include <cstdint>
#include <vector>
#include <cstring>

namespace vit {

#pragma pack(push, 1)
struct Box {
    uint16_t x1, y1, x2, y2; // normalised coords × 65535
    uint8_t  class_id;
    uint8_t  score;           // 0–100
};
#pragma pack(pop)

static_assert(sizeof(Box) == 10, "Box must be 10 bytes");

struct FrameResult {
    uint64_t            timestamp_ms;
    std::vector<Box>    boxes;
};

/// Serialise a FrameResult into a compact binary WebSocket payload.
/// Layout: [uint64 timestamp][uint16 count][Box × N]
inline std::vector<uint8_t> serialise(const FrameResult& result) {
    const uint16_t n = static_cast<uint16_t>(result.boxes.size());
    std::vector<uint8_t> buf(8 + 2 + n * 10);
    uint8_t* p = buf.data();

    std::memcpy(p, &result.timestamp_ms, 8); p += 8;
    std::memcpy(p, &n, 2);                  p += 2;
    for (const auto& box : result.boxes) {
        std::memcpy(p, &box, 10);           p += 10;
    }
    return buf;
}

} // namespace vit
