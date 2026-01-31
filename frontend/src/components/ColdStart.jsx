import { useState } from 'react'

function ColdStart() {
    const [sessionId, setSessionId] = useState('')
    const [questions, setQuestions] = useState([])
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
    const [response, setResponse] = useState('')
    const [recommendations, setRecommendations] = useState([])
    const [complete, setComplete] = useState(false)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const startColdStart = async () => {
        const newSessionId = `session_${Date.now()}`
        setSessionId(newSessionId)
        setLoading(true)
        setError(null)

        try {
            const response = await fetch(`/api/cold-start/init?session_id=${newSessionId}`, {
                method: 'POST'
            })

            if (!response.ok) {
                throw new Error('Failed to initialize cold start')
            }

            const data = await response.json()
            setQuestions(data.questions || [])
            setCurrentQuestionIndex(0)
            setComplete(false)
            setRecommendations([])
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const submitResponse = async () => {
        if (!response.trim()) return

        setLoading(true)
        setError(null)

        try {
            const res = await fetch('/api/cold-start/respond', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    question_index: currentQuestionIndex,
                    response: response
                })
            })

            if (!res.ok) {
                throw new Error('Failed to submit response')
            }

            const data = await res.json()

            if (data.complete) {
                setComplete(true)
                setRecommendations(data.recommendations || [])
            } else {
                setCurrentQuestionIndex(data.question_index)
                setResponse('')
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    if (!sessionId) {
        return (
            <div className="section">
                <h2>Cold Start - New User Onboarding</h2>
                <p style={{ marginBottom: '15px' }}>
                    For users with no purchase history, we use LLM-powered questions to understand preferences
                    and provide personalized recommendations.
                </p>
                <button className="button" onClick={startColdStart} disabled={loading}>
                    {loading ? 'Initializing...' : 'Start Cold Start Flow'}
                </button>
            </div>
        )
    }

    if (complete) {
        return (
            <div className="section">
                <h2>Cold Start Recommendations</h2>
                <p style={{ marginBottom: '15px', color: '#28a745' }}>
                    ✓ Based on your responses, here are personalized recommendations:
                </p>

                {recommendations.length > 0 ? (
                    <div className="product-list">
                        {recommendations.map((rec, index) => (
                            <div key={index} className="product-card">
                                <h3>{index + 1}. {rec.product_name}</h3>
                                <div className="price">£{rec.price?.toFixed(2)}</div>
                                <div className="score">
                                    Category: {rec.category} | Priority: {rec.priority}
                                </div>
                                {rec.reasoning && (
                                    <div className="explanation">
                                        💡 {rec.reasoning}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    <p>No recommendations generated. Please try again.</p>
                )}

                <button
                    className="button"
                    onClick={() => {
                        setSessionId('')
                        setQuestions([])
                        setRecommendations([])
                        setComplete(false)
                    }}
                    style={{ marginTop: '20px' }}
                >
                    Start Over
                </button>
            </div>
        )
    }

    return (
        <div className="section">
            <h2>Cold Start Questions</h2>
            <p style={{ marginBottom: '15px', color: '#666' }}>
                Question {currentQuestionIndex + 1} of {questions.length}
            </p>

            {error && (
                <div className="error">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {questions[currentQuestionIndex] && (
                <div>
                    <div style={{
                        background: '#f8f9fa',
                        padding: '20px',
                        marginBottom: '20px',
                        border: '1px solid #dee2e6',
                        fontSize: '16px'
                    }}>
                        {questions[currentQuestionIndex]}
                    </div>

                    <div className="input-group">
                        <label>Your Response:</label>
                        <input
                            type="text"
                            value={response}
                            onChange={(e) => setResponse(e.target.value)}
                            placeholder="Type your answer here..."
                            onKeyPress={(e) => e.key === 'Enter' && submitResponse()}
                            disabled={loading}
                        />
                    </div>

                    <button
                        className="button"
                        onClick={submitResponse}
                        disabled={loading || !response.trim()}
                    >
                        {loading ? 'Submitting...' :
                            currentQuestionIndex < questions.length - 1 ? 'Next Question' : 'Get Recommendations'}
                    </button>
                </div>
            )}
        </div>
    )
}

export default ColdStart
