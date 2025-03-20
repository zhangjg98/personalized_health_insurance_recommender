import React, { useState, useEffect } from "react";
import { Container, Form, Button, Card, Row, Col, Alert } from "react-bootstrap";
import ReactTooltip from "react-tooltip";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [formData, setFormData] = useState({
    state: "",
    age: "young_adult",
    smoker: "no",
    bmi: "",
    income: "",
    family_size: "",
    chronic_condition: "no",
    medical_care_frequency: "Low",
    preferred_plan_type: ""
  });

  const [recommendation, setRecommendation] = useState(null);
  const [mlSummary, setMlSummary] = useState("");
  const [mlData, setMlData] = useState(null);
  const [outlierMessage, setOutlierMessage] = useState("");
  const [error, setError] = useState("");
  const [tooltip, showTooltip] = useState(true);
  const [feedbackGiven, setFeedbackGiven] = useState(false); // Track if feedback has been given
  const [selectedFeedback, setSelectedFeedback] = useState(null); // Track selected feedback ("Yes" or "No")

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setFeedbackGiven(false); // Reset feedback state when a new recommendation is fetched
    setSelectedFeedback(null);

    try {
      const payload = { ...formData, user_id: 1 }; // Use static user_id for now
      const response = await fetch("/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setRecommendation(data.recommendation);
      setMlSummary(data.ml_summary);
      setMlData(data.ml_prediction);
      setOutlierMessage(data.outlier_message || "");
    } catch (err) {
      setError(err.message);
    }
  };

  const logPlanFeedback = async (rating) => {
    if (feedbackGiven) return; // Prevent multiple feedback submissions

    try {
      const response = await fetch('/log_interaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 1, // Use static user_id for now
          item_id: recommendation?.plan || "Unknown Plan", // Use the plan name as the item_id
          rating,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log(data.message);

      setFeedbackGiven(true); // Mark feedback as given
      setSelectedFeedback(rating === 5 ? "Yes" : "No"); // Set selected feedback
    } catch (err) {
      console.error('Error logging feedback:', err);
    }
  };

  const metricsInfo = [
    {
      key: "Standardized Medicare Payment per Capita",
      tooltip:
        "The average cost per beneficiary after adjusting for regional differences.",
      classificationKey: "Standardized Medicare Payment per Capita Level",
      comparisonKey: "Standardized Medicare Payment per Capita Comparison",
    },
    {
      key: "Average Health Risk Score",
      tooltip:
        "A score representing the overall health risk of the state's Medicare population. Higher scores mean higher expected healthcare needs.",
      classificationKey: "Average Health Risk Score Level",
      comparisonKey: "Average Health Risk Score Comparison",
    },
    {
      key: "Emergency Department Visit Rate (per 1,000 beneficiaries)",
      tooltip:
        "The estimated number of emergency department visits per 1,000 beneficiaries, indicating urgent care utilization.",
      classificationKey: "Emergency Department Visit Rate (per 1,000 beneficiaries) Level",
      comparisonKey: "Emergency Department Visit Rate (per 1,000 beneficiaries) Comparison",
    },
  ];

  const classificationDescriptions = {
    Low: "This value is lower than average.",
    Moderate: "This value is around the average range.",
    High: "This value is higher than average.",
  };

  const classificationColors = {
    Low: "green",
    Moderate: "orange",
    High: "red",
  };

  useEffect(() => {}, [mlData, recommendation, mlSummary, outlierMessage]);

  return (
    <Container className="mt-4">
      <h1 className="mb-4 text-center">Personalized Health Insurance Recommender</h1>
      {error && <Alert variant="danger">{error}</Alert>}
      <Form onSubmit={handleSubmit}>
        <Row>
          <Col md={4}>
            <Form.Group controlId="formAge">
              <Form.Label>Age Range:</Form.Label>
              <Form.Control as="select" name="age" value={formData.age} onChange={handleChange}>
                <option value="young_adult">18-29</option>
                <option value="adult">30-59</option>
                <option value="senior">60+</option>
              </Form.Control>
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group controlId="formSmoker">
              <Form.Label>Are you a smoker?</Form.Label>
              <Form.Control as="select" name="smoker" value={formData.smoker} onChange={handleChange}>
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </Form.Control>
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group controlId="formState">
              <Form.Label>State:</Form.Label>
              <Form.Control as="select" name="state" value={formData.state} onChange={handleChange}>
                <option value="">Prefer not to say</option>
                <option value="AL">Alabama</option>
                <option value="AK">Alaska</option>
                <option value="AZ">Arizona</option>
                <option value="AR">Arkansas</option>
                <option value="CA">California</option>
                <option value="CO">Colorado</option>
                <option value="CT">Connecticut</option>
                <option value="DE">Delaware</option>
                <option value="DC">District of Columbia</option>
                <option value="FL">Florida</option>
                <option value="GA">Georgia</option>
                <option value="HI">Hawaii</option>
                <option value="ID">Idaho</option>
                <option value="IL">Illinois</option>
                <option value="IN">Indiana</option>
                <option value="IA">Iowa</option>
                <option value="KS">Kansas</option>
                <option value="KY">Kentucky</option>
                <option value="LA">Louisiana</option>
                <option value="ME">Maine</option>
                <option value="MD">Maryland</option>
                <option value="MA">Massachusetts</option>
                <option value="MI">Michigan</option>
                <option value="MN">Minnesota</option>
                <option value="MS">Mississippi</option>
                <option value="MO">Missouri</option>
                <option value="MT">Montana</option>
                <option value="NE">Nebraska</option>
                <option value="NV">Nevada</option>
                <option value="NH">New Hampshire</option>
                <option value="NJ">New Jersey</option>
                <option value="NM">New Mexico</option>
                <option value="NY">New York</option>
                <option value="NC">North Carolina</option>
                <option value="ND">North Dakota</option>
                <option value="OH">Ohio</option>
                <option value="OK">Oklahoma</option>
                <option value="OR">Oregon</option>
                <option value="PA">Pennsylvania</option>
                <option value="RI">Rhode Island</option>
                <option value="SC">South Carolina</option>
                <option value="SD">South Dakota</option>
                <option value="TN">Tennessee</option>
                <option value="TX">Texas</option>
                <option value="UT">Utah</option>
                <option value="VT">Vermont</option>
                <option value="VA">Virginia</option>
                <option value="WA">Washington</option>
                <option value="WV">West Virginia</option>
                <option value="WI">Wisconsin</option>
                <option value="WY">Wyoming</option>
                <option value="Territory">U.S. Territories</option>
              </Form.Control>
            </Form.Group>
          </Col>
        </Row>
        <Row>
          <Col md={4}>
            <Form.Group controlId="formBmi">
              <Form.Label>BMI Range (optional):</Form.Label>
              <Form.Control as="select" name="bmi" value={formData.bmi} onChange={handleChange}>
                <option value="">Prefer not to say</option>
                <option value="underweight">(&lt;18.5)</option>
                <option value="normal">(18.5–24.9)</option>
                <option value="overweight">(25.0–29.9)</option>
                <option value="obese">(30+)</option>
              </Form.Control>
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group controlId="formIncome">
              <Form.Label>Annual Income (optional):</Form.Label>
              <Form.Control as="select" name="income" value={formData.income} onChange={handleChange}>
                <option value="">Prefer not to say</option>
                <option value="below_30000">Below $30,000</option>
                <option value="30000_to_74999">$30,000–$74,999</option>
                <option value="75000_to_99999">$75,000–$99,999</option>
                <option value="above_100000">Above $100,000</option>
              </Form.Control>
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group controlId="formFamilySize">
              <Form.Label>Family Size (optional):</Form.Label>
              <Form.Control as="select" name="family_size" value={formData.family_size} onChange={handleChange}>
                <option value="">Prefer not to say</option>
                <option value="1">1 person</option>
                <option value="2_to_3">2–3 people</option>
                <option value="4_plus">4+ people</option>
              </Form.Control>
            </Form.Group>
          </Col>
        </Row>
        <Row>
          <Col md={4}>
            <Form.Group controlId="formChronicCondition">
              <Form.Label>Chronic Condition?</Form.Label>
              <Form.Control as="select" name="chronic_condition" value={formData.chronic_condition} onChange={handleChange}>
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </Form.Control>
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group controlId="formMedicalCareFrequency">
              <Form.Label>How often do you need medical visits?</Form.Label>
              <Form.Control as="select" name="medical_care_frequency" value={formData.medical_care_frequency} onChange={handleChange}>
                <option value="Low">Rarely</option>
                <option value="Moderate">Sometimes</option>
                <option value="High">Often</option>
              </Form.Control>
            </Form.Group>
          </Col>
          <Col md={4}>
            <Form.Group controlId="formPreferredPlanType">
              <Form.Label>Preferred Plan Type:</Form.Label>
              <Form.Control
                as="select"
                name="preferred_plan_type"
                value={formData.preferred_plan_type}
                onChange={handleChange}
              >
                <option value="">Select a plan type</option>
                <option value="HMO">HMO</option>
                <option value="PPO">PPO</option>
                <option value="EPO">EPO</option>
                <option value="POS">POS</option>
              </Form.Control>
            </Form.Group>
          </Col>
        </Row>
        <Button variant="primary" type="submit" className="mt-3">
          Get Recommendation
        </Button>
      </Form>

      {recommendation && (
        <Card className="mt-4">
          <Card.Header>Recommended Plan</Card.Header>
          <Card.Body>
            <Card.Text>{recommendation.plan}</Card.Text>
            <Card.Text><em>{recommendation.justification}</em></Card.Text>
            <div className="mt-3">
              <h5>Was this recommendation helpful?</h5>
              <Button
                variant={selectedFeedback === "Yes" ? "success" : "outline-success"}
                size="sm"
                className={`me-2 ${selectedFeedback === "Yes" ? "selected-feedback" : ""}`}
                onClick={() => logPlanFeedback(5)} // Positive feedback
                disabled={feedbackGiven} // Disable button if feedback is already given
              >
                Yes
              </Button>
              <Button
                variant={selectedFeedback === "No" ? "danger" : "outline-danger"}
                size="sm"
                className={selectedFeedback === "No" ? "selected-feedback" : ""}
                onClick={() => logPlanFeedback(1)} // Negative feedback
                disabled={feedbackGiven} // Disable button if feedback is already given
              >
                No
              </Button>
            </div>
          </Card.Body>
        </Card>
      )}

      {mlSummary && (
        <Card className="mt-4">
          <Card.Header>State-Level Medicare Trends</Card.Header>
          <Card.Body>
            <Card.Text>{mlSummary}</Card.Text>
          </Card.Body>
        </Card>
      )}

      {mlData && mlData.length > 0 && (
        <Card className="mt-4">
          <Card.Header>Predicted Medicare Spending Details</Card.Header>
          <Card.Body>
            <ul>
              {metricsInfo.map((item) => {
                const metricValue = mlData[0][item.key];
                const classification = mlData[0][item.classificationKey];
                const comparison = mlData[0][item.comparisonKey]; // Updated comparison text
                const description = classificationDescriptions[classification] || "No description available.";
                const color = classificationColors[classification] || "black";

                return (
                  <li key={item.key}>
                    <span
                      style={{
                        textDecoration: "underline dotted",
                        cursor: "help",
                      }}
                      data-tip={item.tooltip}
                      onMouseEnter={() => showTooltip(true)} // Show tooltip on hover
                      onMouseLeave={() => {
                        showTooltip(false); // Hide tooltip temporarily
                        setTimeout(() => showTooltip(true), 50); // Re-enable tooltip after a short delay
                      }}
                    >
                      <strong>{item.key}:</strong>
                    </span>{" "}
                    <span style={{ color }}>{metricValue !== undefined ? metricValue.toFixed(4) : "N/A"}</span>{" "}
                    <em>({classification || "Unknown"})</em> - {description}{" "}
                    <span>{comparison}</span> {/* Display comparison as a separate sentence */}
                  </li>
                );
              })}
            </ul>
            {tooltip && <ReactTooltip effect="solid" />} {/* Tooltip component */}
            <p className="mt-3 text-muted">
              <strong>Note:</strong> The values shown in "Predicted Medicare Spending Details" are predictions based on historical data and trained machine learning models. These predictions aim to provide insights into state-level trends and are not exact measurements.
            </p>
            <p className="mt-3 text-muted">
              <strong>Additional Note:</strong> The classification (e.g., "Moderate") is based on thresholds derived from state-level data, specifically the 10th to 90th percentiles. Values within this range are classified as "Moderate". The comparison (e.g., "above the national average") is relative to the national average, calculated as the mean of all state values.
            </p>
          </Card.Body>
        </Card>
      )}

      {outlierMessage && (
        <Card className="mt-4">
          <Card.Header>Outlier Information</Card.Header>
          <Card.Body>
            <Card.Text>{outlierMessage}</Card.Text>
          </Card.Body>
        </Card>
      )}
    </Container>
  );
}

export default App;
