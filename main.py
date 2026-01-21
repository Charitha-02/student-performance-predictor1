import streamlit as st
from streamlit_option_menu import option_menu

from web_pages import home, dataset, analysis, prediction, about, predict_student
from src.logger import logging


def main_application():

    st.set_page_config(
        page_title="Student Performance Analysis and Prediction",
        layout="wide"
    )

    st.markdown("## 🎓 Student Performance Analysis and Prediction")
    st.markdown("---")

    with st.sidebar:
        bar = option_menu(
            menu_title="Menu",
            menu_icon="list",
            options=[
                "Home",
                "About Dataset",
                "Data Analysis",
                "Prediction Model",
                "Student Predict",
                "About"
            ]
        )

    if bar == "Home":
        home.home()

    elif bar == "About Dataset":
        dataset.dataset()

    elif bar == "Data Analysis":
        analysis.data_analysis()

    elif bar == "Student Predict":
        predict_student.student_predict()

    elif bar == "Prediction Model":
        prediction.model_prediction()

    elif bar == "About":
        about.about()


def main():
    main_application()


if __name__ == "__main__":
    main()
