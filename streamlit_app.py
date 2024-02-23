# <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRst3OatJS2Ajr7kPxQIcs19JWf7SN7bclcuK8h4pOvcSBpZeo7p-kGErc6aX67uFnomVetgHk-s6hx/embed?start=false&loop=false&delayms=3000" frameborder="0" width="1440" height="839" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
import streamlit as st
import streamlit.components.v1 as components

st.title("ONDC Index Presentation URL")
st.image("images/perf_banner.svg")  
components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vRst3OatJS2Ajr7kPxQIcs19JWf7SN7bclcuK8h4pOvcSBpZeo7p-kGErc6aX67uFnomVetgHk-s6hx/embed?start=false&loop=false&delayms=3000",width=1440, height=839, scrolling=False)
st.text("Time to complete 10,000 searches ~ 90s")
st.text("Roughly 6667 Searches per minute on a single instance")
st.image("images/10000-requests.gif", width=1280)
st.image("images/response_times.svg")