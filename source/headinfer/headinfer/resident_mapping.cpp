#include <cstdint>
#include <vector>

// Build the five attention-order index arrays in one pass. Position is a
// bounded token index, so direct addressing avoids a search per selected row.
template <typename HitIndex>
int map_rows(
    const int64_t* const* previous, const int64_t* const* current,
    const int64_t* previous_lengths, const int64_t* current_lengths,
    const int64_t* entries, int64_t groups, int64_t capacity,
    HitIndex* hit_sources, HitIndex* hit_destinations,
    int64_t* miss_rows, int64_t* miss_destinations,
    int64_t* history_destinations, int64_t* hit_count,
    int64_t* miss_count, bool previous_attention_layout) {
  if (capacity <= 0 || groups <= 0) return 1;
  std::vector<int64_t> lookup(static_cast<size_t>(capacity), -1);
  int64_t hits = 0, misses = 0, previous_offset = 0, attention_offset = 0;
  for (int64_t group = 0; group < groups; ++group) {
    const auto previous_size = previous_lengths[group];
    const auto current_size = current_lengths[group];
    if (previous_size < 0 || current_size < 0 || entries[group] < 0) return 2;
    for (int64_t row = 0; row < previous_size; ++row) {
      const auto position = previous[group][row];
      if (position < 0 || position >= capacity) return 3;
      lookup[position] = row;
    }
    for (int64_t row = 0; row < current_size; ++row) {
      const auto position = current[group][row];
      if (position < 0 || position >= capacity) return 4;
      const auto source = lookup[position];
      const auto destination = attention_offset + row;
      history_destinations[attention_offset - group + row] = destination;
      if (source >= 0) {
        if constexpr (sizeof(HitIndex) == 4) {
          if (previous_offset + source > INT32_MAX || destination > INT32_MAX)
            return 5;
        }
        hit_sources[hits] = previous_offset + source;
        hit_destinations[hits++] = destination;
      } else {
        miss_rows[misses] = entries[group] * capacity + position;
        miss_destinations[misses++] = destination;
      }
    }
    for (int64_t row = 0; row < previous_size; ++row)
      lookup[previous[group][row]] = -1;
    previous_offset += previous_size + (previous_attention_layout ? 1 : 0);
    attention_offset += current_size + 1;
  }
  *hit_count = hits;
  *miss_count = misses;
  return 0;
}

extern "C" int map_resident_rows(
    const int64_t* const* previous, const int64_t* const* current,
    const int64_t* previous_lengths, const int64_t* current_lengths,
    const int64_t* entries, int64_t groups, int64_t capacity,
    int64_t* hit_sources, int64_t* hit_destinations,
    int64_t* miss_rows, int64_t* miss_destinations,
    int64_t* history_destinations, int64_t* hit_count,
    int64_t* miss_count) {
  return map_rows(previous, current, previous_lengths, current_lengths,
                  entries, groups, capacity, hit_sources, hit_destinations,
                  miss_rows, miss_destinations, history_destinations,
                  hit_count, miss_count, false);
}

extern "C" int map_resident_rows_compact_hits(
    const int64_t* const* previous, const int64_t* const* current,
    const int64_t* previous_lengths, const int64_t* current_lengths,
    const int64_t* entries, int64_t groups, int64_t capacity,
    int32_t* hit_sources, int32_t* hit_destinations,
    int64_t* miss_rows, int64_t* miss_destinations,
    int64_t* history_destinations, int64_t* hit_count,
    int64_t* miss_count) {
  return map_rows(previous, current, previous_lengths, current_lengths,
                  entries, groups, capacity, hit_sources, hit_destinations,
                  miss_rows, miss_destinations, history_destinations,
                  hit_count, miss_count, false);
}

extern "C" int map_resident_rows_compact_attention_layout(
    const int64_t* const* previous, const int64_t* const* current,
    const int64_t* previous_lengths, const int64_t* current_lengths,
    const int64_t* entries, int64_t groups, int64_t capacity,
    int32_t* hit_sources, int32_t* hit_destinations,
    int64_t* miss_rows, int64_t* miss_destinations,
    int64_t* history_destinations, int64_t* hit_count,
    int64_t* miss_count) {
  return map_rows(previous, current, previous_lengths, current_lengths,
                  entries, groups, capacity, hit_sources, hit_destinations,
                  miss_rows, miss_destinations, history_destinations,
                  hit_count, miss_count, true);
}
