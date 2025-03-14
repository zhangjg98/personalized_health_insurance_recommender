import React, { useState, useEffect } from "react";
import ReactTooltip from "react-tooltip";

function App() {
  const [formData, setFormData] = useState({
    state: "",
    age: "",
    smoker: "",
    bmi: "",
    income: "",
    family_size: "",
    chronic_condition: "",
    medical_care_frequency: "",
  });

  const [recommendation, setRecommendation] = useState(null);
  const [mlSummary, setMlSummary] = useState("");
  const [mlData, setMlData] = useState(null);
  const [error, setError] = useState("");
  const [tooltip, showTooltip] = useState(true); // Tooltip visibility state

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setRecommendation(data.rule_recommendation);
      setMlSummary(data.ml_summary);
      setMlData(data.ml_data);
    } catch (err) {
      setError(err.message);
    }
  };

  // Define key metrics and tooltips
  const metricsInfo = [
    {
      key: "Standardized Medicare Payment per Capita",
      tooltip:
        "The average cost per beneficiary after adjusting for regional differences.",
    },
    {
      key: "Average Health Risk Score",
      tooltip:
        "A score representing the overall health risk of the state's Medicare population. Higher scores mean higher expected healthcare needs.",
    },
    {
      key: "Emergency Department Visit Rate (per 1,000 beneficiaries)",
      tooltip:
        "The estimated number of emergency department visits per 1,000 beneficiaries, indicating urgent care utilization.",
    },
  ];

  useEffect(() => {
    ReactTooltip.rebuild(); // Reinitialize tooltips on component update
  }, [mlData]);

  return (
    <div className="App">
      <h1>Personalized Health Insurance Recommender</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Age Range:
          <select name="age" value={formData.age} onChange={handleChange}>
            <option value="young_adult">18-29</option>
            <option value="adult">30-59</option>
            <option value="senior">60+</option>
          </select>
        </label>
        <br />

        <label>
          Are you a smoker?:
          <select name="smoker" value={formData.smoker} onChange={handleChange}>
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
        </label>
        <br />

        <label>
          State:
          <select name="state" value={formData.state} onChange={handleChange}>
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
            {/* ... */}
          </select>
        </label>
        <br />

        {/* Additional fields for BMI, income, family size, etc. */}
        <label>
          BMI Range (optional):
          <select name="bmi" value={formData.bmi} onChange={handleChange}>
            <option value="">Prefer not to say</option>
            <option value="underweight">(&lt;18.5)</option>
            <option value="normal">(18.5–24.9)</option>
            <option value="overweight">(25.0–29.9)</option>
            <option value="obese">(30+)</option>
          </select>
        </label>
        <br />

        <label>
          Annual Income (optional):
          <select name="income" value={formData.income} onChange={handleChange}>
            <option value="">Prefer not to say</option>
            <option value="below_30000">Below $30,000</option>
            <option value="30000_to_74999">$30,000–$74,999</option>
            <option value="75000_to_99999">$75,000–$99,999</option>
            <option value="above_100000">Above $100,000</option>
          </select>
        </label>
        <br />

        <label>
          Family Size (optional):
          <select name="family_size" value={formData.family_size} onChange={handleChange}>
            <option value="">Prefer not to say</option>
            <option value="1">1 person</option>
            <option value="2_to_3">2–3 people</option>
            <option value="4_plus">4+ people</option>
          </select>
        </label>
        <br />

        <label>
          Chronic Condition?:
          <select name="chronic_condition" value={formData.chronic_condition} onChange={handleChange}>
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
        </label>
        <br />

        <label>
          How often do you need medical visits?:
          <select name="medical_care_frequency" value={formData.medical_care_frequency} onChange={handleChange}>
            <option value="Low">Rarely</option>
            <option value="Moderate">Sometimes</option>
            <option value="High">Often</option>
          </select>
        </label>
        <br />

        <button type="submit">Get Recommendation</button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {recommendation && (
        <div>
          <h2>Recommended Plan</h2>
          <p>{recommendation.plan}</p>
          <p>
            <em>{recommendation.justification}</em>
          </p>
        </div>
      )}

      {mlSummary && (
        <div>
          <h2>State-Level Medicare Trends</h2>
          <p>{mlSummary}</p>
        </div>
      )}

      {mlData && (
        <div>
          <h3>Predicted Medicare Spending Details</h3>
          <ul>
            {metricsInfo.map((item) => (
              <li key={item.key}>
                <span
                  style={{
                    textDecoration: "underline dotted",
                    cursor: "help",
                  }}
                  data-tip={item.tooltip}
                  data-for="tooltip"
                  onMouseEnter={() => showTooltip(true)}
                  onMouseLeave={() => {
                    showTooltip(false);
                    setTimeout(() => showTooltip(true), 50);
                  }}
                >
                  <strong>{item.key}:</strong>
                </span>{" "}
                {mlData[item.key]}
              </li>
            ))}
          </ul>
          {tooltip && (
            <ReactTooltip
              id="tooltip"
              place="right"
              type="dark"
              effect="solid"
              delayHide={100}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;
