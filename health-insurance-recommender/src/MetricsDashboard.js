import React, { useState, useEffect } from "react";
import { Container, Row, Col, Card, Alert } from "react-bootstrap";

function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch("/metrics");
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        setMetrics(data);
      } catch (err) {
        setError(err.message);
      }
    };

    fetchMetrics();
  }, []);

  return (
    <Container className="mt-4">
      <h2 className="mb-4 text-center">Model Evaluation Metrics</h2>
      {error && <Alert variant="danger">{error}</Alert>}
      {metrics && (
        <Row>
          {Object.entries(metrics).map(([key, value]) => (
            <Col md={3} key={key}>
              <Card className="mb-4">
                <Card.Body>
                  <Card.Title>{key}</Card.Title>
                  <Card.Text>{value.toFixed(4)}</Card.Text>
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      )}
    </Container>
  );
}

export default MetricsDashboard;
