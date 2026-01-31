import { useState, useEffect } from 'react'

function Recommendations({ userId }) {
    const [recommendations, setRecommendations] = useState([])
    const [insight, setInsight] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [processingTime, setProcessingTime] = useState(null)

    useEffect(() => {
        if (userId) {
            fetchRecommendations()
            fetchInsight()
        }
    }, [userId])

    const fetchRecommendations = async () => {
        setLoading(true)
        setError(null)

        try {
            const response = await fetch(`/api/recommendations/${userId}?top_k=10&include_explanations=true`)

            if (!response.ok) {
                throw new Error('Failed to fetch recommendations')
            }

            const data = await response.json()
            setRecommendations(data.recommendations || [])
            setProcessingTime(data.processing_time_ms)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchInsight = async () => {
        try {
            const response = await fetch(`/api/recommendations/${userId}/insight`)

            if (response.ok) {
                const data = await response.json()
                setInsight(data)
            }
        } catch (err) {
            console.error('Failed to fetch insight:', err)
        }
    }

    if (!userId) {
        return (
            <div className="section">
                <h2>Personalized Recommendations</h2>
                <p>Please enter a User ID above to see recommendations.</p>
                <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
                    Try user IDs like: 12346, 12347, 12348, etc.
                </p>
            </div>
        )
    }

    return (
        <div>
            {insight && !insight.is_new_user && (
                <div className="insight-box">
                    <h3 style={{ marginBottom: '10px' }}>User Insight</h3>
                    <p>{insight.insight}</p>

                    <div className="stats" style={{ marginTop: '15px' }}>
                        <div className="stat-card">
                            <div className="label">Total Spend</div>
                            <div className="value">£{insight.total_spend?.toFixed(2) || 0}</div>
                        </div>
                        <div className="stat-card">
                            <div className="label">Purchases</div>
                            <div className="value">{insight.purchase_count || 0}</div>
                        </div>
                        <div className="stat-card">
                            <div className="label">Top Categories</div>
                            <div className="value" style={{ fontSize: '14px' }}>
                                {insight.top_categories?.join(', ') || 'N/A'}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="section">
                <h2>Top 10 Recommendations</h2>

                {loading && <div className="loading">Loading recommendations...</div>}

                {error && (
                    <div className="error">
                        <strong>Error:</strong> {error}
                    </div>
                )}

                {!loading && !error && recommendations.length === 0 && (
                    <p>No recommendations available. User may not exist or models not trained.</p>
                )}

                {!loading && recommendations.length > 0 && (
                    <>
                        <div className="product-list">
                            {recommendations.map((rec, index) => (
                                <div key={index} className="product-card">
                                    <h3>
                                        {index + 1}. {rec.product_name}
                                    </h3>
                                    <div className="price">£{rec.price?.toFixed(2)}</div>
                                    <div className="score">
                                        Match Score: {(rec.score * 100).toFixed(1)}% |
                                        Category: {rec.category} |
                                        Method: {rec.method}
                                    </div>
                                    {rec.explanation && (
                                        <div className="explanation">
                                            💡 {rec.explanation}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>

                        {processingTime !== null && (
                            <div className="processing-time">
                                ⚡ Processing time: {processingTime.toFixed(2)}ms
                                {processingTime < 500 ? ' ✓ (Target: <500ms)' : ' ⚠ (Exceeds 500ms target)'}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}

export default Recommendations
