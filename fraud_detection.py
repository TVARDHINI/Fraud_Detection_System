"""
Advanced Fraud Detection System with Machine Learning
Supports multiple ML algorithms and real-time transaction analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.random_forest = None
        self.label_encoders = {}
        self.user_profile = {
            'avg_transaction': 2500,
            'max_transaction': 15000,
            'common_locations': ['Mumbai', 'Hyderabad', 'Bangalore'],
            'common_categories': ['Groceries', 'Restaurants', 'Fuel', 'Online Shopping'],
            'common_merchants': ['Big Bazaar', 'Amazon India', 'Indian Oil', 'Swiggy', 'Flipkart'],
            'typical_hours': list(range(9, 21))
        }
        
    def generate_sample_data(self, n_samples=1000, fraud_ratio=0.15):
        """Generate synthetic transaction data for training"""
        print(f"Generating {n_samples} sample transactions...")
        
        transactions = []
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        categories = ['Groceries', 'Restaurants', 'Fuel', 'Online Shopping', 
                     'Electronics', 'Travel', 'Healthcare', 'Entertainment']
        locations = ['Mumbai', 'Hyderabad', 'Bangalore', 'Delhi', 'Chennai', 
                    'Kolkata', 'Pune', 'Dubai', 'Singapore', 'London']
        merchants = ['Big Bazaar', 'Amazon India', 'Indian Oil', 'Swiggy', 
                    'Flipkart', 'Reliance Digital', 'DMart', 'Zomato', 'PayTM', 'PhonePe']
        
        # Generate normal transactions
        for i in range(n_normal):
            hour = random.choice(self.user_profile['typical_hours'])
            transactions.append({
                'transaction_id': f'TXN{i:06d}',
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 90), 
                                                       hours=hour, 
                                                       minutes=random.randint(0, 59)),
                'amount': random.uniform(500, 8000),
                'merchant': random.choice(self.user_profile['common_merchants']),
                'category': random.choice(self.user_profile['common_categories']),
                'location': random.choice(self.user_profile['common_locations']),
                'card_last4': '4532',
                'is_fraud': 0
            })
        
        # Generate fraudulent transactions
        for i in range(n_fraud):
            hour = random.choice([1, 2, 3, 4, 5, 22, 23])  # Unusual hours
            transactions.append({
                'transaction_id': f'TXN{n_normal + i:06d}',
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 90), 
                                                       hours=hour, 
                                                       minutes=random.randint(0, 59)),
                'amount': random.uniform(25000, 100000),  # High amounts
                'merchant': random.choice(merchants),
                'category': random.choice(categories),
                'location': random.choice(['Dubai', 'Singapore', 'London', 'Delhi']),  # Unusual locations
                'card_last4': '4532',
                'is_fraud': 1
            })
        
        df = pd.DataFrame(transactions)
        random.shuffle(transactions)
        df = pd.DataFrame(transactions)
        
        print(f"✓ Generated {len(df)} transactions ({n_normal} normal, {n_fraud} fraud)")
        return df
    
    def feature_engineering(self, df):
        """Extract features from transaction data"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_unusual_hour'] = (~df['hour'].isin(self.user_profile['typical_hours'])).astype(int)
        
        # Amount-based features
        df['amount_deviation'] = (df['amount'] - self.user_profile['avg_transaction']) / self.user_profile['avg_transaction']
        df['exceeds_max'] = (df['amount'] > self.user_profile['max_transaction']).astype(int)
        
        # Location features
        df['is_common_location'] = df['location'].isin(self.user_profile['common_locations']).astype(int)
        
        # Category features
        df['is_common_category'] = df['category'].isin(self.user_profile['common_categories']).astype(int)
        
        # Merchant features
        df['is_common_merchant'] = df['merchant'].isin(self.user_profile['common_merchants']).astype(int)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for ML models"""
        df = self.feature_engineering(df)
        
        # Encode categorical variables
        categorical_cols = ['merchant', 'category', 'location']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
            
            # Handle unseen labels by replacing them with 'Unknown'
            df[col] = df[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown')
            
            # Add 'Unknown' to encoder if not present
            if 'Unknown' not in self.label_encoders[col].classes_:
                self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
            
            df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Select features for modeling
        feature_cols = ['amount', 'hour', 'day_of_week', 'is_weekend', 'is_unusual_hour',
                       'amount_deviation', 'exceeds_max', 'is_common_location',
                       'is_common_category', 'is_common_merchant',
                       'merchant_encoded', 'category_encoded', 'location_encoded']
        
        X = df[feature_cols]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        return X, y, df
    
    def train_models(self, df):
        """Train multiple ML models"""
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        X, y, df_processed = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest (Unsupervised Anomaly Detection)
        print("\n1. Training Isolation Forest...")
        self.isolation_forest = IsolationForest(contamination=0.15, random_state=42)
        self.isolation_forest.fit(X_train_scaled)
        if_predictions = self.isolation_forest.predict(X_test_scaled)
        if_predictions = [1 if x == -1 else 0 for x in if_predictions]  # Convert to 0/1
        
        print("   Isolation Forest Results:")
        print(f"   Accuracy: {accuracy_score(y_test, if_predictions):.2%}")
        
        # Train Random Forest (Supervised Classification)
        print("\n2. Training Random Forest Classifier...")
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.random_forest.fit(X_train_scaled, y_train)
        rf_predictions = self.random_forest.predict(X_test_scaled)
        
        print("   Random Forest Results:")
        print(f"   Accuracy: {accuracy_score(y_test, rf_predictions):.2%}")
        print("\n   Classification Report:")
        print(classification_report(y_test, rf_predictions, target_names=['Normal', 'Fraud']))
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.random_forest.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n   Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "="*60)
        print("✓ Model training completed!")
        print("="*60)
        
        return X_test, y_test, rf_predictions
    
    def predict_fraud(self, transaction_data):
        """Predict if a single transaction is fraudulent"""
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
        
        X, _, df_processed = self.prepare_data(df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        if_pred = self.isolation_forest.predict(X_scaled)[0]
        if_score = 1 if if_pred == -1 else 0
        
        rf_pred = self.random_forest.predict(X_scaled)[0]
        rf_proba = self.random_forest.predict_proba(X_scaled)[0][1]
        
        # Combine predictions
        is_fraud = (if_score == 1 or rf_pred == 1)
        fraud_probability = rf_proba
        
        # Determine status
        if fraud_probability >= 0.7:
            status = 'BLOCKED'
        elif fraud_probability >= 0.4:
            status = 'FLAGGED'
        else:
            status = 'SAFE'
        
        # Get risk factors
        risk_factors = []
        if df_processed['amount'].values[0] > self.user_profile['max_transaction']:
            risk_factors.append('Amount exceeds normal maximum')
        if df_processed['is_unusual_hour'].values[0]:
            risk_factors.append('Transaction at unusual time')
        if not df_processed['is_common_location'].values[0]:
            risk_factors.append('Unusual location detected')
        if not df_processed['is_common_category'].values[0]:
            risk_factors.append('Unusual spending category')
        
        result = {
            'transaction_id': df['transaction_id'].values[0] if 'transaction_id' in df.columns else 'N/A',
            'status': status,
            'fraud_probability': fraud_probability,
            'is_fraud': is_fraud,
            'risk_factors': risk_factors,
            'models': {
                'isolation_forest': 'Anomaly' if if_score == 1 else 'Normal',
                'random_forest': 'Fraud' if rf_pred == 1 else 'Normal'
            }
        }
        
        return result
    
    def analyze_transaction(self, transaction):
        """Analyze and display a single transaction"""
        result = self.predict_fraud(transaction)
        
        print("\n" + "="*60)
        print("TRANSACTION ANALYSIS")
        print("="*60)
        print(f"Transaction ID: {transaction.get('transaction_id', 'N/A')}")
        print(f"Amount: ₹{transaction['amount']:,.2f}")
        print(f"Merchant: {transaction['merchant']}")
        print(f"Category: {transaction['category']}")
        print(f"Location: {transaction['location']}")
        print(f"Time: {transaction.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        print(f"Status: {result['status']}")
        print(f"Fraud Probability: {result['fraud_probability']:.1%}")
        print(f"\nModel Predictions:")
        print(f"  - Isolation Forest: {result['models']['isolation_forest']}")
        print(f"  - Random Forest: {result['models']['random_forest']}")
        
        if result['risk_factors']:
            print(f"\nRisk Factors Detected:")
            for factor in result['risk_factors']:
                print(f"  ⚠ {factor}")
        
        print("="*60)
        
        return result


def main():
    """Main function to demonstrate the fraud detection system"""
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "ML-BASED FRAUD DETECTION SYSTEM" + " "*16 + "║")
    print("║" + " "*15 + "Powered by Machine Learning" + " "*16 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # Initialize system
    fds = FraudDetectionSystem()
    
    # Generate training data
    df = fds.generate_sample_data(n_samples=1000, fraud_ratio=0.15)
    
    # Train models
    fds.train_models(df)
    
    # Test with sample transactions
    print("\n\n" + "╔" + "="*58 + "╗")
    print("║" + " "*16 + "TESTING WITH SAMPLE TRANSACTIONS" + " "*11 + "║")
    print("╚" + "="*58 + "╝")
    
    # Test transaction 1: Normal transaction
    test_txn1 = {
        'transaction_id': 'TEST001',
        'timestamp': datetime.now(),
        'amount': 1500,
        'merchant': 'Big Bazaar',
        'category': 'Groceries',
        'location': 'Mumbai',
        'card_last4': '4532'
    }
    
    print("\n[TEST 1: Normal Transaction]")
    fds.analyze_transaction(test_txn1)
    
    # Test transaction 2: Suspicious transaction
    test_txn2 = {
        'transaction_id': 'TEST002',
        'timestamp': datetime.now().replace(hour=2),
        'amount': 45000,
        'merchant': 'Unknown Merchant',
        'category': 'Electronics',
        'location': 'Dubai',
        'card_last4': '4532'
    }
    
    print("\n[TEST 2: Suspicious Transaction]")
    fds.analyze_transaction(test_txn2)
    
    # Test transaction 3: Borderline case
    test_txn3 = {
        'transaction_id': 'TEST003',
        'timestamp': datetime.now().replace(hour=21),
        'amount': 12000,
        'merchant': 'Flipkart',
        'category': 'Online Shopping',
        'location': 'Bangalore',
        'card_last4': '4532'
    }
    
    print("\n[TEST 3: Borderline Transaction]")
    fds.analyze_transaction(test_txn3)
    
    print("\n\n" + "="*60)
    print("FRAUD DETECTION SYSTEM DEMO COMPLETED")
    print("="*60)
    


if __name__ == "__main__":
    main()
