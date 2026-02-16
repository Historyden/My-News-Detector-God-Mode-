/* CSS Styling for main card and prediction box */
.main-card {
    border: 1px solid #ccc;
    border-radius: 10px;
    padding: 20px;
    margin: 10px;
    background-color: #f9f9f9;
}

.prediction-box {
    margin-top: 10px;
    padding: 15px;
}

.prediction-box.fake {
    background-color: #ffcccc;
    border: 1px solid #ff0000;
}

.prediction-box.real {
    background-color: #ccffcc;
    border: 1px solid #009900;
}

.confidence-bar {
    width: 100%;
    background-color: #eee;
    border-radius: 5px;
}

.confidence-fill {
    height: 100%;
    border-radius: 5px;
}

.reasoning-box {
    background-color: #f0f0f0;
    border-radius: 5px;
    padding: 10px;
    margin-top: 10px;
}

.reasoning-item {
    margin: 5px 0;
}

.leaderboard-item {
    display: flex;
    justify-content: space-between;
    padding: 10px;
}

.leaderboard-rank.gold {
    color: gold;
}

.leaderboard-rank.silver {
    color: silver;
}

.leaderboard-rank.bronze {
    color: #cd7f32;
}

.suspicious {
    background-color: #ffeb3b;
}

.button {
    padding: 10px 15px;
    margin: 5px;
    border: none;
    border-radius: 5px;
    background-color: #007bff;
    color: white;
    cursor: pointer;
}

.button:hover {
    background-color: #0056b3;
    transition: background-color 0.3s;
} 
