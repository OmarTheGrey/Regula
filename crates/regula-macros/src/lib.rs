//! REGULA Macros - Procedural macros for the REGULA framework.
//!
//! This crate provides the `#[derive(GraphState)]` macro for automatically
//! implementing the `GraphState` trait on structs.
//!
//! # Usage
//!
//! ```ignore
//! use regula_macros::GraphState;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, GraphState, Serialize, Deserialize)]
//! struct MyState {
//!     messages: Vec<String>,
//!     
//!     #[reducer(append)]
//!     history: Vec<String>,
//! }
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Error, Fields, Ident,
    Result as SynResult,
};

/// Channel type specification from attributes.
#[derive(Debug, Clone, PartialEq)]
enum ChannelType {
    /// Default: last value semantics
    LastValue,
    /// Append reducer for Vec types
    Append,
    /// Add reducer for numeric types
    Add,
    /// Ephemeral: cleared after each step
    Ephemeral,
    /// Any value: last writer wins
    AnyValue,
    /// Custom reducer function
    Custom(String),
}

impl Default for ChannelType {
    fn default() -> Self {
        Self::LastValue
    }
}

/// Parse a single field and extract its channel configuration.
struct FieldConfig {
    name: Ident,
    channel_type: ChannelType,
}

impl FieldConfig {
    fn from_field(field: &syn::Field) -> SynResult<Option<Self>> {
        let name = match &field.ident {
            Some(ident) => ident.clone(),
            None => return Ok(None), // Skip unnamed fields (tuple structs)
        };

        let mut channel_type = ChannelType::default();

        // Process attributes
        for attr in &field.attrs {
            if attr.path().is_ident("reducer") {
                channel_type = Self::parse_reducer_attr(attr)?;
            } else if attr.path().is_ident("channel") {
                channel_type = Self::parse_channel_attr(attr)?;
            }
        }

        Ok(Some(Self { name, channel_type }))
    }

    fn parse_reducer_attr(attr: &Attribute) -> SynResult<ChannelType> {
        let mut result = ChannelType::LastValue;
        
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("append") {
                result = ChannelType::Append;
            } else if meta.path.is_ident("add") {
                result = ChannelType::Add;
            } else {
                // Custom reducer function name
                let fn_name = meta
                    .path
                    .get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "custom".to_string());
                result = ChannelType::Custom(fn_name);
            }
            Ok(())
        })?;

        Ok(result)
    }

    fn parse_channel_attr(attr: &Attribute) -> SynResult<ChannelType> {
        let mut result = ChannelType::LastValue;

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("ephemeral") {
                result = ChannelType::Ephemeral;
            } else if meta.path.is_ident("last_value") {
                result = ChannelType::LastValue;
            } else if meta.path.is_ident("any_value") {
                result = ChannelType::AnyValue;
            } else if meta.path.is_ident("append") {
                result = ChannelType::Append;
            } else if meta.path.is_ident("add") {
                result = ChannelType::Add;
            } else {
                return Err(meta.error(
                    "unknown channel type. Use: ephemeral, last_value, any_value, append, or add",
                ));
            }
            Ok(())
        })?;

        Ok(result)
    }

    fn to_channel_spec_tokens(&self) -> TokenStream2 {
        match &self.channel_type {
            ChannelType::LastValue => {
                quote! { regula_core::ChannelSpec::LastValue }
            }
            ChannelType::Append => {
                quote! { regula_core::ChannelSpec::Reducer(regula_core::channel::ReducerType::Append) }
            }
            ChannelType::Add => {
                quote! { regula_core::ChannelSpec::Reducer(regula_core::channel::ReducerType::Add) }
            }
            ChannelType::Ephemeral => {
                quote! { regula_core::ChannelSpec::Ephemeral }
            }
            ChannelType::AnyValue => {
                quote! { regula_core::ChannelSpec::AnyValue }
            }
            ChannelType::Custom(name) => {
                quote! { regula_core::ChannelSpec::Reducer(regula_core::channel::ReducerType::Custom(#name.to_string())) }
            }
        }
    }
}

/// Derive macro for implementing `GraphState` on structs.
///
/// This macro generates the `channels()` method based on the struct fields
/// and any `#[reducer(...)]` or `#[channel(...)]` attributes.
///
/// # Attributes
///
/// - `#[reducer(append)]`: Use append reducer for Vec fields
/// - `#[reducer(add)]`: Use add reducer for numeric fields
/// - `#[reducer(fn_name)]`: Use a custom reducer function
/// - `#[channel(ephemeral)]`: Mark field as ephemeral (cleared each step)
/// - `#[channel(last_value)]`: Use last value semantics (default)
/// - `#[channel(any_value)]`: Allow multiple writes, last writer wins
///
/// # Example
///
/// ```ignore
/// #[derive(Clone, GraphState, Serialize, Deserialize)]
/// struct AgentState {
///     /// Messages in the conversation (last value semantics)
///     messages: Vec<Message>,
///     
///     /// Tool call history (appended across steps)
///     #[reducer(append)]
///     tool_calls: Vec<ToolCall>,
///     
///     /// Running total
///     #[reducer(add)]
///     total: i32,
///     
///     /// Temporary scratch space (cleared each step)
///     #[channel(ephemeral)]
///     scratch: Option<String>,
/// }
/// ```
#[proc_macro_derive(GraphState, attributes(reducer, channel))]
pub fn derive_graph_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match derive_graph_state_impl(input) {
        Ok(tokens) => TokenStream::from(tokens),
        Err(err) => TokenStream::from(err.to_compile_error()),
    }
}

fn derive_graph_state_impl(input: DeriveInput) -> SynResult<TokenStream2> {
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Ensure it's a struct with named fields
    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => &fields_named.named,
            Fields::Unnamed(_) => {
                return Err(Error::new_spanned(
                    &input.ident,
                    "GraphState can only be derived for structs with named fields",
                ));
            }
            Fields::Unit => {
                return Err(Error::new_spanned(
                    &input.ident,
                    "GraphState cannot be derived for unit structs",
                ));
            }
        },
        Data::Enum(_) => {
            return Err(Error::new_spanned(
                &input.ident,
                "GraphState can only be derived for structs, not enums",
            ));
        }
        Data::Union(_) => {
            return Err(Error::new_spanned(
                &input.ident,
                "GraphState can only be derived for structs, not unions",
            ));
        }
    };

    // Parse each field
    let mut field_configs = Vec::new();
    for field in fields {
        if let Some(config) = FieldConfig::from_field(field)? {
            field_configs.push(config);
        }
    }

    // Generate channel insertions
    let channel_insertions: Vec<TokenStream2> = field_configs
        .iter()
        .map(|config| {
            let field_name = config.name.to_string();
            let channel_spec = config.to_channel_spec_tokens();
            quote! {
                channels.insert(#field_name.to_string(), #channel_spec);
            }
        })
        .collect();

    // Generate field names for field_names() method
    let field_name_literals: Vec<TokenStream2> = field_configs
        .iter()
        .map(|config| {
            let field_name = config.name.to_string();
            quote! { #field_name }
        })
        .collect();

    let expanded = quote! {
        impl #impl_generics regula_core::GraphState for #name #ty_generics #where_clause {
            fn channels() -> std::collections::HashMap<String, regula_core::ChannelSpec> {
                let mut channels = std::collections::HashMap::new();
                #(#channel_insertions)*
                channels
            }

            fn field_names() -> Vec<&'static str> {
                vec![#(#field_name_literals),*]
            }
        }
    };

    Ok(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_type_default() {
        assert_eq!(ChannelType::default(), ChannelType::LastValue);
    }
}
