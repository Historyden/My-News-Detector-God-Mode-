st.markdown("""
<style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    .main-card {
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        margin: 20px 0;
    }
    
    .prediction-box.fake {
        background: linear-gradient(135deg, rgba(255,75,75,0.1) 0%, rgba(255,75,75,0.05) 100%);
        border-left: 5px solid #ff4b4b;
    }
    
    .prediction-box.real {
        background: linear-gradient(135deg, rgba(0,210,106,0.1) 0%, rgba(0,210,106,0.05) 100%);
        border-left: 5px solid #00d26a;
    }
    
    .prediction-label {
        font-size: 1.8em;
        font-weight: 700;
        margin: 10px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 25px;
        margin: 15px 0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-fill {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.9em;
        transition: width 0.3s ease;
    }
    
    .confidence-fill.fake {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff7b7b 100%);
    }
    
    .confidence-fill.real {
        background: linear-gradient(90deg, #00d26a 0%, #00ff8a 100%);
    }
    
    .reasoning-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .reasoning-item {
        padding: 10px 0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .reasoning-item:last-child {
        border-bottom: none;
    }
    
    .leaderboard-item {
        display: flex;
        align-items: center;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .leaderboard-item:hover {
        transform: translateX(5px);
    }
    
    .leaderboard-rank {
        font-size: 1.5em;
        font-weight: 700;
        min-width: 50px;
        text-align: center;
    }
    
    .leaderboard-rank.gold {
        color: #ffd700;
    }
    
    .leaderboard-rank.silver {
        color: #c0c0c0;
    }
    
    .leaderboard-rank.bronze {
        color: #cd7f32;
    }
    
    .suspicious {
        background: linear-gradient(120deg, #ffff00, #ffff00);
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)