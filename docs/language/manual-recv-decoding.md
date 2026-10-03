# Manual recv argument decoding

Use `#[decode(manual)]` on a recv arm when its handler reads the input through
checked calldata views or a custom decoder:

```fe
use std::abi::DynArray
use std::evm::CallData

msg Requests {
    #[selector = sol("first(uint256[])")]
    First { values: DynArray<u256> } -> u256,
}

pub contract Reader {
    recv Requests {
        #[decode(manual)]
        First { .. } -> u256 {
            CallData::with_base(4).array_view<u256>(0).get(0)
        }
    }
}
```

The message still defines the selector, typed caller encoding, JSON ABI inputs,
and return type. The compiler preserves the payment check and return encoding.
It skips argument decoding for this arm, including the usual argument validation.
The handler must perform all required frame, offset, length, and value checks.
For example, `ArrayView` validates its array frame and validates element contents
as they are accessed; unused dynamic tails remain unexamined.

The arm must use an unbound record pattern such as `First { .. }`. Binding or
matching individual fields would require decoded values and is rejected. This
attribute applies to recv arms, including payable arms; it does not apply to
constructors or fallback arms. Arms without the attribute retain normal argument
decoding even when all fields are ignored.
