const { useState } = React;

function MathAgentApp() {
    const [question, setQuestion] = useState('');
    const [solution, setSolution] = useState(null);
    const [loading, setLoading] = useState(false);
    const [feedback, setFeedback] = useState({rating: 5, comment: ''});

    const solveProblem = async () => {
        if (!question.trim()) return;
        
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/solve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            });
            const data = await response.json();
            setSolution(data);
        } catch (error) {
            alert('Error: Make sure the backend is running on port 8000.');
        }
        setLoading(false);
    };

    const submitFeedback = async () => {
        if (!solution) return;
        try {
            await fetch('http://localhost:8000/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    question: solution.question,
                    solution: solution.solution,
                    rating: feedback.rating,
                    feedback: feedback.comment
                })
            });
            alert('Feedback submitted!');
            setFeedback({rating: 5, comment: ''});
        } catch (error) {
            alert('Error submitting feedback');
        }
    };

    return (
        <div className="container">
            <h1>🎯 Math Routing Agent</h1>
            <p>Ask any mathematics question and get step-by-step solutions</p>
            
            <div>
                <input 
                    type="text" 
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="e.g., Solve x² - 5x + 6 = 0"
                    onKeyPress={(e) => e.key === 'Enter' && solveProblem()}
                />
                <button onClick={solveProblem} disabled={loading}>
                    {loading ? 'Solving...' : 'Solve'}
                </button>
            </div>

            {solution && (
                <div className="solution">
                    <h3>Solution (Source: {solution.source})</h3>
                    <pre style={{whiteSpace: 'pre-wrap'}}>{solution.solution}</pre>
                    
                    <div className="feedback">
                        <h4>Rate this solution:</h4>
                        <select 
                            value={feedback.rating}
                            onChange={(e) => setFeedback({...feedback, rating: parseInt(e.target.value)})}
                        >
                            <option value="5">Excellent</option>
                            <option value="4">Good</option>
                            <option value="3">Average</option>
                            <option value="2">Poor</option>
                            <option value="1">Bad</option>
                        </select>
                        <input 
                            type="text" 
                            placeholder="Comments..."
                            value={feedback.comment}
                            onChange={(e) => setFeedback({...feedback, comment: e.target.value})}
                            style={{marginLeft: '10px', padding: '5px', width: '200px'}}
                        />
                        <button onClick={submitFeedback} style={{marginLeft: '10px'}}>
                            Submit Feedback
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

ReactDOM.render(<MathAgentApp />, document.getElementById('root'));