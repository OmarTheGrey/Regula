//! Integration tests for the GraphState derive macro.

use regula_core::{GraphState, ChannelSpec};
use regula_core::channel::ReducerType;
use regula_macros::GraphState as DeriveGraphState;
use serde::{Deserialize, Serialize};

/// Basic state with default LastValue channels
#[derive(Clone, Default, Serialize, Deserialize, DeriveGraphState)]
struct BasicState {
    counter: i32,
    name: String,
}

#[test]
fn test_basic_state_channels() {
    let channels = BasicState::channels();
    assert_eq!(channels.len(), 2);
    
    // Both should be LastValue by default
    assert!(matches!(channels.get("counter"), Some(ChannelSpec::LastValue)));
    assert!(matches!(channels.get("name"), Some(ChannelSpec::LastValue)));
}

#[test]
fn test_basic_state_field_names() {
    let fields = BasicState::field_names();
    assert_eq!(fields.len(), 2);
    assert!(fields.contains(&"counter"));
    assert!(fields.contains(&"name"));
}

/// State with reducer channels
#[derive(Clone, Default, Serialize, Deserialize, DeriveGraphState)]
struct ReducerState {
    #[reducer(append)]
    messages: Vec<String>,
    
    #[reducer(add)]
    total: i32,
}

#[test]
fn test_reducer_state_channels() {
    let channels = ReducerState::channels();
    assert_eq!(channels.len(), 2);
    
    // Check messages channel
    let messages_channel = channels.get("messages");
    assert!(messages_channel.is_some());
    assert!(matches!(messages_channel, Some(ChannelSpec::Reducer(ReducerType::Append))));
    
    // Check total channel
    let total_channel = channels.get("total");
    assert!(total_channel.is_some());
    assert!(matches!(total_channel, Some(ChannelSpec::Reducer(ReducerType::Add))));
}

/// State with channel attributes
#[derive(Clone, Default, Serialize, Deserialize, DeriveGraphState)]
struct ChannelState {
    #[channel(ephemeral)]
    scratch: Option<String>,
    
    #[channel(last_value)]
    explicit_last: i32,
    
    #[channel(any_value)]
    optional: Option<bool>,
}

#[test]
fn test_channel_state_channels() {
    let channels = ChannelState::channels();
    assert_eq!(channels.len(), 3);
    
    // Check scratch channel (ephemeral)
    let scratch_channel = channels.get("scratch");
    assert!(matches!(scratch_channel, Some(ChannelSpec::Ephemeral)));
    
    // Check explicit_last channel (last_value)
    let explicit_last_channel = channels.get("explicit_last");
    assert!(matches!(explicit_last_channel, Some(ChannelSpec::LastValue)));
    
    // Check optional channel (any_value)
    let optional_channel = channels.get("optional");
    assert!(matches!(optional_channel, Some(ChannelSpec::AnyValue)));
}

/// State with custom reducer
#[derive(Clone, Default, Serialize, Deserialize, DeriveGraphState)]
struct CustomReducerState {
    #[reducer(my_custom_merge)]
    data: Vec<u8>,
}

#[test]
fn test_custom_reducer() {
    let channels = CustomReducerState::channels();
    assert_eq!(channels.len(), 1);
    
    let data_channel = channels.get("data");
    assert!(data_channel.is_some());
    if let Some(ChannelSpec::Reducer(ReducerType::Custom(name))) = data_channel {
        assert_eq!(name, "my_custom_merge");
    } else {
        panic!("Expected Custom reducer");
    }
}

/// Mixed state combining all attribute types
#[derive(Clone, Default, Serialize, Deserialize, DeriveGraphState)]
struct MixedState {
    // Default: LastValue
    count: u32,
    
    // Reducer: Append
    #[reducer(append)]
    items: Vec<i32>,
    
    // Channel: Ephemeral
    #[channel(ephemeral)]
    temp_data: Option<String>,
}

#[test]
fn test_mixed_state_channels() {
    let channels = MixedState::channels();
    assert_eq!(channels.len(), 3);
    
    // Check count (default LastValue)
    assert!(matches!(channels.get("count"), Some(ChannelSpec::LastValue)));
    
    // Check items (Reducer Append)
    assert!(matches!(
        channels.get("items"), 
        Some(ChannelSpec::Reducer(ReducerType::Append))
    ));
    
    // Check temp_data (Ephemeral)
    assert!(matches!(channels.get("temp_data"), Some(ChannelSpec::Ephemeral)));
}
