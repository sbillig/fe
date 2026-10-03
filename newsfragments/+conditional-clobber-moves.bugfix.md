Fix false move conflicts when releasing fresh native buffers after growing independent buffers. Conditional raw-pointer corruption now carries physical separation requirements through ownership checking.
Recursive functions updating independent owned buffers now keep finite, separate output identities and conservative raw-corruption conditions.
