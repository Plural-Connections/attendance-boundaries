"""
Landing page for streamlit app; includes access code prompt/box if necessary
"""

import streamlit as st
import streamlit_modal as modal
import streamlit.components.v1 as components
from streamlit2 import include_markdown

import referral_codes


def referral_code_entered():
    """Checks whether a referral code entered by the user is correct."""

    # TODO: update this to actually check against possible referral codes
    st.session_state["referral_code_tried"] = True
    if referral_codes.is_referral_code_valid(st.session_state["referral_code_input"]):
        st.session_state["referral_code_valid"] = True
        st.session_state["referral_code"] = st.session_state["referral_code_input"].replace(")", "")
    else:
        st.session_state["referral_code_valid"] = False


def check_consent():
    """Returns `True` if the user provides consent and also"""

    query_params = st.experimental_get_query_params()
    referral_code = None
    if "referral_code" in query_params:
        referral_code = query_params["referral_code"][0]
        if referral_codes.is_referral_code_valid(referral_code):
            st.session_state["referral_code_valid"] = True
            st.session_state["referral_code"] = referral_code.replace(")","")

    consent_text_placeholder = st.empty()
    access_code_entry_placeholder = st.empty()
    continue_button_placeholder = st.empty()
    read_more_button_placeholder = st.empty()

    with consent_text_placeholder:
        include_markdown("couhes_message")

    if not st.session_state["referral_code_valid"]:
        with access_code_entry_placeholder.container():

            if not "referral_code_tried" in st.session_state:
                # First run, show input for referral code
                st.text_input(
                    "Referral code",
                    on_change=referral_code_entered,
                    key="referral_code_input",
                )
                st.session_state["referral_code_skipped"] = st.checkbox(
                    "Continue without an access code"
                )

            elif (
                st.session_state["referral_code_tried"]
                and not st.session_state["referral_code_valid"]
            ):

                st.text_input(
                    "Referral code",
                    on_change=referral_code_entered,
                    key="referral_code_input",
                )
                st.error(
                    "Referral code invalid.  Please try again, or check below to continue without an access code."
                )
                st.session_state["referral_code_skipped"] = st.checkbox(
                    "Continue without an access code"
                )
            else:
                # Code correct
                st.session_state["referral_code_valid"] = True

    if "skip_consent" in query_params:
        st.session_state["continue_pressed"] = True
    else:
        with continue_button_placeholder:
            st.session_state["continue_pressed"] = st.button("Continue")

    with read_more_button_placeholder.container():
        with st.expander("Learn more"):
            st.write("\n\n\n")
            include_markdown("more_about_project")

    return [
        consent_text_placeholder,
        access_code_entry_placeholder,
        continue_button_placeholder,
        read_more_button_placeholder,
    ]
