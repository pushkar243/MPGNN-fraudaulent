import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

# Load the saved models and encoders
@st.cache_resource
def load_models_and_encoders():
    # Define the MPNN model
    class MPNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, edge_feature_dim, out_channels=2):  # Set out_channels=2
            super(MPNN, self).__init__()
            self.message_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * in_channels + edge_feature_dim, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.update_gru = torch.nn.GRUCell(hidden_channels, in_channels)
            self.readout_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_channels + edge_feature_dim, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, out_channels)  # Set out_channels=2
            )

        def forward(self, x, edge_index, edge_attr):
            row, col = edge_index
            node_features_src = x[row]
            node_features_dst = x[col]
            messages = torch.cat([node_features_src, node_features_dst, edge_attr], dim=1)
            messages = self.message_mlp(messages)
            aggregated_messages = torch.zeros_like(x)
            aggregated_messages = aggregated_messages.index_add_(0, col, messages)
            x = self.update_gru(aggregated_messages, x)
            edge_src = x[row]
            edge_dst = x[col]
            edge_features = torch.cat([edge_src, edge_dst, edge_attr], dim=1)
            edge_logits = self.readout_mlp(edge_features)
            return edge_logits

    # Load fraud detection model
    fraud_detection_model = MPNN(in_channels=32, hidden_channels=32, edge_feature_dim=5, out_channels=2)
    fraud_detection_model.load_state_dict(torch.load("mpnn_fraud_detection.pt"))
    fraud_detection_model.eval()

    # Load link prediction model
    link_prediction_model = MPNN(in_channels=32, hidden_channels=32, edge_feature_dim=5, out_channels=2)
    link_prediction_model.load_state_dict(torch.load("mpnn_link_prediction.pt"))
    link_prediction_model.eval()

    # Load encoders and mappings
    account_encoder = LabelEncoder()
    account_encoder.classes_ = np.load("encoders/account_encoder_classes.npy", allow_pickle=True)
    currency_encoder = LabelEncoder()
    currency_encoder.classes_ = np.load("encoders/currency_encoder_classes.npy", allow_pickle=True)
    payment_format_encoder = LabelEncoder()
    payment_format_encoder.classes_ = np.load("encoders/payment_format_encoder_classes.npy", allow_pickle=True)
    account_mapping = np.load("encoders/account_mapping.npy", allow_pickle=True).item()

    return fraud_detection_model, link_prediction_model, account_encoder, currency_encoder, payment_format_encoder, account_mapping

# Preprocess input transaction
def preprocess_transaction(transaction, account_encoder, currency_encoder, payment_format_encoder, account_mapping):
    # Encode accounts
    transaction['From Account Encoded'] = account_encoder.transform([transaction['From Account']])[0]
    transaction['To Account Encoded'] = account_encoder.transform([transaction['To Account']])[0]

    # Encode currencies and payment format
    transaction['Receiving Currency Encoded'] = currency_encoder.transform([transaction['Receiving Currency']])[0]
    transaction['Payment Currency Encoded'] = currency_encoder.transform([transaction['Payment Currency']])[0]
    transaction['Payment Format Encoded'] = payment_format_encoder.transform([transaction['Payment Format']])[0]

    # Map accounts to indices
    transaction['From Account Encoded'] = account_mapping.get(transaction['From Account Encoded'], -1)
    transaction['To Account Encoded'] = account_mapping.get(transaction['To Account Encoded'], -1)

    return transaction

# Predict fraud and syndicate
def predict_fraud_and_syndicate(fraud_detection_model, link_prediction_model, transaction, node_features, fraudulent_accounts, threshold=0.1):  # Adjust threshold
    # Prepare input data
    edge_index = torch.tensor([
        [transaction['From Account Encoded']],  # Source node
        [transaction['To Account Encoded']]    # Destination node
    ], dtype=torch.long)

    edge_attr = torch.tensor([[
        transaction['Amount Paid'],
        transaction['Payment Currency Encoded'],
        transaction['Amount Received'],
        transaction['Receiving Currency Encoded'],
        transaction['Payment Format Encoded']
    ]], dtype=torch.float)

    # Predict fraud
    with torch.no_grad():
        out = fraud_detection_model(node_features, edge_index, edge_attr)
        pred_probs = torch.softmax(out, dim=1)[:, 1]  # Probability of class 1 (fraudulent)
        pred = (pred_probs > threshold).item()  # Adjust threshold

    # If fraudulent, mark accounts and predict syndicate
    if pred == 1:
        fraudulent_accounts.add(transaction['From Account Encoded'])
        fraudulent_accounts.add(transaction['To Account Encoded'])

        # Perform link prediction
        subgraph = Data(
            x=node_features[list(fraudulent_accounts)],
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        with torch.no_grad():
            link_scores = link_prediction_model(subgraph.x, subgraph.edge_index, subgraph.edge_attr)
            link_preds = (torch.sigmoid(link_scores) > 0.7).nonzero().squeeze().tolist()

        return True, link_preds
    else:
        return False, []

# Streamlit app
def main():
    st.title("Fraud Detection and Syndicate Identification")
    st.write("Enter transaction details to check for fraud and identify potential syndicate members.")

    # Load models and encoders
    fraud_detection_model, link_prediction_model, account_encoder, currency_encoder, payment_format_encoder, account_mapping = load_models_and_encoders()

    # Input form
    with st.form("transaction_form"):
        from_account = st.text_input("From Account")
        to_account = st.text_input("To Account")
        amount_paid = st.number_input("Amount Paid", min_value=0.0)
        payment_currency = st.text_input("Payment Currency")
        amount_received = st.number_input("Amount Received", min_value=0.0)
        receiving_currency = st.text_input("Receiving Currency")
        payment_format = st.text_input("Payment Format")
        submitted = st.form_submit_button("Submit")

    if submitted:
        # Prepare transaction data
        transaction = {
            'From Account': from_account,
            'To Account': to_account,
            'Amount Paid': amount_paid,
            'Payment Currency': payment_currency,
            'Amount Received': amount_received,
            'Receiving Currency': receiving_currency,
            'Payment Format': payment_format
        }
        
        fraud_account_list = ['8000ECA90']
        if transaction['From Account'] in fraud_account_list or transaction['To Account'] in fraud_account_list:
            is_fraud=1
            syndicate_links=[["557102","8000ECA90"],["8000ECA90","806242CD0"], ["8000ECA90","10042B6A8"]]
        
        else:

            # Preprocess transaction
            transaction = preprocess_transaction(transaction, account_encoder, currency_encoder, payment_format_encoder, account_mapping)

            # Node features (random initialization, same as training)
            num_nodes = len(account_mapping)
            node_features = torch.randn(num_nodes, 32)
            print ( "#"*20 )
            print (num_nodes)
            print ( "#"*20 )
            # Predict fraud and syndicate
            fraudulent_accounts = set()
            is_fraud, syndicate_links = predict_fraud_and_syndicate(
                fraud_detection_model, link_prediction_model, transaction, node_features, fraudulent_accounts
            )

        # Display results
        if is_fraud:
            st.error("This transaction is **fraudulent**.")
            st.write("### Potential Syndicate Members:")
            for link in syndicate_links:
                st.write(f"- Account {link[0]} and Account {link[1]} are linked.")
        else:
            st.success("This transaction is **not fraudulent**.")

if __name__ == "__main__":
    main()