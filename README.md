# Intelligent Ad Inventory Scoring and Adaptive Bidding Optimization

Welcome to the GitHub repository for the Intelligent Ad Inventory Scoring and Adaptive Bidding Optimization project. This repository is dedicated to addressing two pivotal challenges in the digital advertising space: enhancing the evaluation of ad inventory quality and optimizing real-time bidding strategies to maximize ROI.

## Overview

This project is at the intersection of cutting-edge technology and digital marketing, focusing on the development of advanced algorithms to navigate the complexities of the RTB marketplace efficiently. It aims to strategically allocate advertising budgets to maximize engagement and conversion rates, adhering to budgetary constraints and Key Performance Indicators (KPIs).

### Part 1: Retrieval-Augmented Generation (RAG) for Inventory Scoring

The project utilizes a Retrieval-Augmented Generation model to sophisticatedly assess the quality of ad inventory by incorporating a wealth of historical and real-time data. This method is designed to provide a granular and accurate scoring system, enabling informed bidding decisions.

#### Highlights:

- **Vector Database Creation:** Building a detailed vector database with multi-dimensional representations of campaign and inventory data.
- **Inventory Retrieval:** Using advanced queries to identify high-potential inventory groups based on historical campaign similarities.
- **Performance Forecast Scoring with CDN:** Leveraging a Cross and Deep Network for predictive modeling of inventory performance, forecasting their effectiveness in achieving campaign goals.
- **External Data Integration:** Enhancing scoring accuracy through the analysis of hosting platform content, adding a layer of semantic understanding to the inventory scoring process.

### Part 2: Adaptive Bidding Optimizer

The second part introduces an adaptive algorithm for real-time bidding strategy optimization. This component focuses on accurately predicting bid factors for each ad inventory, facilitating efficient budget utilization while striving for maximum engagement.

#### Objectives:

- **Bid Factor Prediction:** Calculating optimal budget allocations for individual inventory bids.
- **Effectiveness Evaluation:** Rigorously assessing the algorithm's cost efficiency, KPI achievement, and profitability.
- **Strategy Adjustment:** Dynamically modifying bidding strategies based on inventory scores and prevailing market conditions.
- **Data Utilization and Innovation:** Tackling the challenge of limited data on lost bids to enhance bidding strategy refinement.

## Repository Structure

- `notebooks/`: Jupyter notebooks for detailed EDA, feature engineering, and model exploration.
- `rag/`: Implementation of the Retrieval-Augmented Generation model, including the integration of Cross and Deep Network for performance forecasting.
- `scripts/`: Contains helper scripts and utilities to support model development and data processing.
- `tests/`: Scripts for testing code integrity and functionality.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository:**
   ```
   git clone https://github.com/<your-username>/Intelligent-Ad-Inventory-Scoring-and-Adaptive-Bidding-Optimization.git
   ```
2. **Environment Setup:**
   ```
   cd Intelligent-Ad-Inventory-Scoring-and-Adaptive-Bidding-Optimization
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Explore the Notebooks:**
   ```
   jupyter notebook
   ```


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
