import { useState } from 'react'

function Search({ userId }) {
    const [query, setQuery] = useState('')
    const [results, setResults] = useState([])
    const [queryUnderstanding, setQueryUnderstanding] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [processingTime, setProcessingTime] = useState(null)

    const handleSearch = async () => {
        if (!query.trim()) return

        setLoading(true)
        setError(null)

        try {
            const response = await fetch('/api/search/natural', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    user_id: userId || null,
                    top_k: 10
                })
            })

            if (!response.ok) {
                throw new Error('Search failed')
            }

            const data = await response.json()
            setResults(data.results || [])
            setQueryUnderstanding(data.query_understanding)
            setProcessingTime(data.processing_time_ms)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const exampleQueries = [
        "christmas decorations under £10",
        "kitchen items for cooking",
        "party supplies and decorations",
        "garden accessories"
    ]

    return (
        <div className="section">
            <h2>Natural Language Search</h2>
            <p style={{ marginBottom: '15px', color: '#666' }}>
                Search using natural language. The system will understand your intent, extract filters, and find relevant products.
            </p>

            <div className="search-box">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g., 'Show me kitchen items under £20'"
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                />
                <button onClick={handleSearch} disabled={loading}>
                    {loading ? 'Searching...' : 'Search'}
                </button>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <p style={{ fontSize: '13px', color: '#666', marginBottom: '8px' }}>Try these examples:</p>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {exampleQueries.map((ex, i) => (
                        <button
                            key={i}
                            onClick={() => setQuery(ex)}
                            style={{
                                padding: '6px 12px',
                                background: '#f0f0f0',
                                border: '1px solid #ddd',
                                cursor: 'pointer',
                                fontSize: '12px'
                            }}
                        >
                            {ex}
                        </button>
                    ))}
                </div>
            </div>

            {queryUnderstanding && (
                <div style={{
                    background: '#f8f9fa',
                    padding: '15px',
                    marginBottom: '20px',
                    border: '1px solid #dee2e6'
                }}>
                    <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Query Understanding:</h3>
                    <p style={{ fontSize: '13px' }}>
                        <strong>Intent:</strong> {queryUnderstanding.intent || 'General search'}<br />
                        {queryUnderstanding.category && <><strong>Category:</strong> {queryUnderstanding.category}<br /></>}
                        {queryUnderstanding.max_price && <><strong>Max Price:</strong> £{queryUnderstanding.max_price}<br /></>}
                        {queryUnderstanding.features?.length > 0 && (
                            <><strong>Features:</strong> {queryUnderstanding.features.join(', ')}</>
                        )}
                    </p>
                </div>
            )}

            {error && (
                <div className="error">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {loading && <div className="loading">Searching...</div>}

            {!loading && results.length > 0 && (
                <>
                    <h3 style={{ marginBottom: '15px' }}>Search Results ({results.length})</h3>
                    <div className="product-list">
                        {results.map((result, index) => (
                            <div key={index} className="product-card">
                                <h3>{index + 1}. {result.product_name}</h3>
                                <div className="price">£{result.price?.toFixed(2)}</div>
                                <div className="score">
                                    Relevance: {(result.relevance_score * 100).toFixed(1)}% |
                                    Category: {result.category}
                                </div>
                                {result.explanation && (
                                    <div className="explanation">
                                        💡 {result.explanation}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    {processingTime !== null && (
                        <div className="processing-time">
                            ⚡ Processing time: {processingTime.toFixed(2)}ms
                            {processingTime < 2000 ? ' ✓ (Target: <2000ms)' : ' ⚠ (Exceeds 2000ms target)'}
                        </div>
                    )}
                </>
            )}

            {!loading && results.length === 0 && query && (
                <p>No results found. Try a different query or check if models are trained.</p>
            )}
        </div>
    )
}

export default Search
