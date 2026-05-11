import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("Employee Attrition Prediction")
st.write(
	"This app follows the logistic-regression workflow from `pract.ipynb` and uses only the four key satisfaction inputs from the notebook: EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance, and RelationshipSatisfaction."
)


@st.cache_data
def load_data() -> pd.DataFrame:
	return pd.read_csv("Hr.csv")


@st.cache_resource
def train_model(data: pd.DataFrame):
	feature_columns = [
		"EnvironmentSatisfaction",
		"JobSatisfaction",
		"WorkLifeBalance",
		"RelationshipSatisfaction",
	]
	working_data = data.loc[:, feature_columns + ["Attrition"]].copy()
	working_data["Attrition"] = working_data["Attrition"].map({"Yes": 1, "No": 0})

	features = working_data[feature_columns]
	target = working_data["Attrition"]

	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=32,
	)

	model = LogisticRegression(max_iter=1000)
	model.fit(x_train, y_train)
	predictions = model.predict(x_test)

	metrics = {
		"accuracy": accuracy_score(y_test, predictions),
		"precision": precision_score(y_test, predictions, zero_division=0),
		"recall": recall_score(y_test, predictions, zero_division=0),
		"f1": f1_score(y_test, predictions, zero_division=0),
		"confusion_matrix": confusion_matrix(y_test, predictions),
		"classification_report": classification_report(y_test, predictions, zero_division=0),
	}

	return model, feature_columns, metrics


data = load_data()
model, feature_columns, metrics = train_model(data)

with st.sidebar:
	st.header("Dataset")
	st.metric("Rows", f"{data.shape[0]:,}")
	st.metric("Columns", f"{data.shape[1]:,}")
	st.caption("Preview of the raw data used by the notebook.")
	st.dataframe(data.head(), use_container_width=True)


left, right = st.columns([1.1, 0.9])

with left:
	st.subheader("Model performance")
	metric_cols = st.columns(4)
	metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
	metric_cols[1].metric("Precision", f"{metrics['precision']:.3f}")
	metric_cols[2].metric("Recall", f"{metrics['recall']:.3f}")
	metric_cols[3].metric("F1", f"{metrics['f1']:.3f}")

	st.text("Confusion matrix")
	confusion_df = pd.DataFrame(
		metrics["confusion_matrix"],
		index=["Actual No", "Actual Yes"],
		columns=["Predicted No", "Predicted Yes"],
	)
	st.dataframe(confusion_df, use_container_width=True)

	st.text("Classification report")
	st.code(metrics["classification_report"], language="text")

with right:
	st.subheader("Predict attrition")
	st.caption("Fill in the four satisfaction scores, then run a prediction with the trained logistic regression model.")

	with st.form("prediction_form"):
		input_values: dict[str, object] = {}

		for column in feature_columns:
			input_values[column] = st.slider(
				column,
				min_value=1,
				max_value=4,
				value=2,
				step=1,
			)

		submitted = st.form_submit_button("Predict")

	if submitted:
		prediction_frame = pd.DataFrame([input_values])
		prediction_frame = prediction_frame[feature_columns]

		predicted_class = model.predict(prediction_frame)[0]
		predicted_probability = model.predict_proba(prediction_frame)[0][1]

		if predicted_class == 1:
			st.error(f"Predicted attrition: Yes ({predicted_probability:.1%} probability)")
		else:
			st.success(f"Predicted attrition: No ({(1 - predicted_probability):.1%} probability)")

# st.subheader("Raw data preview")
# st.dataframe(data, use_container_width=True)

