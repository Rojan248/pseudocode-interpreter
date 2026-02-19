## 2024-05-23 - Symbol Table Caching
**Learning:** Interpreter performance is heavily dominated by variable lookup in loops. Moving from O(depth) to O(1) lookup using a cache (hash map of stacks) yielded ~21% wall-clock speedup for a simple loop.
**Action:** Always look for O(N) operations in hot paths (like variable access) and consider trading memory for O(1) access.
