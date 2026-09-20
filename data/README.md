# Dataset

This project uses synthetically generated food-delivery operational data for
machine learning and analytics demonstration.

The complete datasets are intentionally excluded from this repository because
of their file size.

## Dataset Generation

The full dataset can be recreated using the data-generation pipeline:

```bash
py src/data_generator.py

This generates the delivery-order dataset used for:
- ETA regression
- Delay classification
- Feature engineering
- SQL analytics
- Model evaluation
- Streamlit dashboard analysis
Dataset Structure
The generated data includes:
- Delivery/order information
- Restaurant characteristics
- Courier information
- Distance and traffic conditions
- Weather conditions
- Kitchen preparation time
- Delivery duration
- SLA/delay information
The project uses a chronological train/validation/test split to prevent
look-ahead data leakage.
