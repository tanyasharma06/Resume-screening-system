import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import the CSS file

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/upload", formData);
      setResult(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h2>Resume Screening & Job Matching</h2>
      
      <div className="upload-box">
        <input type="file" id="fileInput" onChange={handleFileChange} />
        <label htmlFor="fileInput" className="file-label">
          {file ? file.name : "Choose a Resume"}
        </label>
        <button className="upload-btn" onClick={handleUpload} disabled={loading}>
          {loading ? "Uploading..." : "Upload & Analyze"}
        </button>
      </div>

      {result && (
        <div className="result-box">
          <h3>Results:</h3>
          <p><strong>Match Score:</strong> {result.match_score}%</p>
          <p><strong>Skill Match:</strong> {result.skill_match}%</p>
          <p><strong>Matched Skills:</strong> {result.matched_skills.join(", ")}</p>
          <p><strong>Missing Skills:</strong> {result.missing_skills.join(", ")}</p>
        </div>
      )}
    </div>
  );
}

export default App;


