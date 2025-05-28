import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict

class ProductRecommender:
    def __init__(self):
        # Load product data (in a real app, this would come from a database)
        self.products = self.load_product_data()
        
        # Initialize user-item interaction matrix (empty)
        self.user_item_matrix = defaultdict(dict)
        
        # Preprocess data and build recommendation models
        self.prepare_recommendation_models()
    
    def load_product_data(self):
        """Load product data with attributes for content-based filtering"""
        products = {
            # Sample product data - in reality this would come from your database
            "p1": {
                "id": "p1",
                "name": "Jaipur Blue Pottery Vase",
                "category": "Handicrafts",
                "region": "Rajasthan",
                "price": 2499,
                "tags": ["pottery", "blue", "handmade", "home decor", "traditional"],
                "description": "Traditional blue pottery from Jaipur with intricate designs"
            },
            "p2": {
                "id": "p2",
                "name": "Banarasi Silk Saree",
                "category": "Textiles",
                "region": "Uttar Pradesh",
                "price": 5999,
                "tags": ["saree", "silk", "handwoven", "traditional wear", "bridal"],
                "description": "Authentic Banarasi silk saree with zari work"
            },
            "p3": {
                "id": "p3",
                "name": "Brass Spice Box (Masala Dabba)",
                "category": "Home Decor",
                "region": "Rajasthan",
                "price": 1799,
                "tags": ["kitchen", "brass", "handcrafted", "traditional", "utensil"],
                "description": "Traditional Indian spice box made of brass with compartments"
            },
            "p4": {
                "id": "p4",
                "name": "Kashmiri Pashmina Shawl",
                "category": "Textiles",
                "region": "Kashmir",
                "price": 4299,
                "tags": ["shawl", "pashmina", "winter", "luxury", "handmade"],
                "description": "Genuine Kashmiri pashmina shawl with embroidery"
            },
            "p5": {
                "id": "p5",
                "name": "Handcarved Wooden Chess Set",
                "category": "Handicrafts",
                "region": "Kerala",
                "price": 3499,
                "tags": ["chess", "wooden", "handcarved", "games", "decor"],
                "description": "Traditional Indian wooden chess set with intricate carvings"
            }
        }
        return products
    
    def prepare_recommendation_models(self):
        """Prepare content-based recommendation models"""
        # Create a DataFrame from products
        product_list = list(self.products.values())
        self.df_products = pd.DataFrame(product_list)
        
        # Combine text features for content-based filtering
        self.df_products['combined_features'] = self.df_products.apply(
            lambda x: f"{x['category']} {x['region']} {' '.join(x['tags'])} {x['description']}", 
            axis=1
        )
        
        # Create TF-IDF matrix
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df_products['combined_features'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create a reverse map of indices to product IDs
        self.indices = pd.Series(self.df_products.index, index=self.df_products['id']).to_dict()
    
    def record_user_interaction(self, user_id, product_id, interaction_type='view', rating=None):
        """
        Record user interactions with products
        interaction_type can be: 'view', 'cart', 'purchase', 'rating'
        """
        if user_id not in self.user_item_matrix:
            self.user_item_matrix[user_id] = {}
            
        if interaction_type == 'rating' and rating is not None:
            self.user_item_matrix[user_id][product_id] = rating
        else:
            # Default weights for different interaction types
            weights = {'view': 1, 'cart': 3, 'purchase': 5}
            if interaction_type in weights:
                current_score = self.user_item_matrix[user_id].get(product_id, 0)
                self.user_item_matrix[user_id][product_id] = current_score + weights[interaction_type]
    
    def get_content_based_recommendations(self, product_id, num_recommendations=5):
        """Get content-based recommendations for a product"""
        idx = self.indices[product_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the most similar products
        sim_scores = sim_scores[1:num_recommendations+1]
        
        # Get the product indices
        product_indices = [i[0] for i in sim_scores]
        
        # Return the top recommended products
        return self.df_products['id'].iloc[product_indices].tolist()
    
    def get_user_recommendations(self, user_id, num_recommendations=5):
        """Get personalized recommendations for a user"""
        # If user has no interactions, return popular items
        if user_id not in self.user_item_matrix or not self.user_item_matrix[user_id]:
            return self.get_popular_products(num_recommendations)
        
        # For users with interactions, combine collaborative and content-based filtering
        user_products = list(self.user_item_matrix[user_id].keys())
        
        # Get recommendations based on all user's interacted products
        recommended_products = set()
        for product_id in user_products:
            content_recs = self.get_content_based_recommendations(product_id, num_recommendations)
            recommended_products.update(content_recs)
        
        # Remove products user has already interacted with
        recommended_products = [p for p in recommended_products if p not in user_products]
        
        # If not enough recommendations, add popular products
        if len(recommended_products) < num_recommendations:
            additional_needed = num_recommendations - len(recommended_products)
            popular_recs = self.get_popular_products(additional_needed)
            recommended_products.extend(popular_recs)
        
        return recommended_products[:num_recommendations]
    
    def get_popular_products(self, num_recommendations=5):
        """Get currently popular products (could be based on sales, views, etc.)"""
        # In a real app, this would come from analytics data
        # Here we just return a static list of "popular" items
        return ["p2", "p4", "p1", "p5", "p3"][:num_recommendations]
    
    def get_recommendations_for_homepage(self, user_id=None):
        """
        Get recommendations for the homepage
        - For logged-in users: personalized recommendations
        - For guests: popular items + regional specialties
        """
        if user_id and user_id in self.user_item_matrix:
            # Personalized recommendations for logged-in users
            personalized = self.get_user_recommendations(user_id, 3)
            other_recommendations = {
                "popular": self.get_popular_products(3),
                "regional": self.get_products_by_region("Rajasthan", 3)  # Default region
            }
            return {
                "personalized": personalized,
                **other_recommendations
            }
        else:
            # For non-logged-in users
            return {
                "popular": self.get_popular_products(4),
                "handicrafts": self.get_products_by_category("Handicrafts", 4),
                "textiles": self.get_products_by_category("Textiles", 4),
                "regional": self.get_products_by_region("Kashmir", 4)
            }
    
    def get_products_by_category(self, category, num_recommendations=5):
        """Get products by category"""
        products = self.df_products[self.df_products['category'] == category]
        if len(products) > num_recommendations:
            products = products.sample(num_recommendations)
        return products['id'].tolist()
    
    def get_products_by_region(self, region, num_recommendations=5):
        """Get products by region"""
        products = self.df_products[self.df_products['region'] == region]
        if len(products) > num_recommendations:
            products = products.sample(num_recommendations)
        return products['id'].tolist()

# Example usage
if __name__ == "__main__":
    recommender = ProductRecommender()
    
    # Simulate some user interactions
    recommender.record_user_interaction("user1", "p1", "view")
    recommender.record_user_interaction("user1", "p1", "cart")
    recommender.record_user_interaction("user1", "p3", "purchase")
    recommender.record_user_interaction("user1", "p4", "view")
    
    # Get recommendations
    print("Recommendations for user1:")
    print(recommender.get_user_recommendations("user1"))
    
    print("\nContent-based recommendations for product p1:")
    print(recommender.get_content_based_recommendations("p1"))
    
    print("\nHomepage recommendations for user1:")
    print(recommender.get_recommendations_for_homepage("user1"))
    
    print("\nHomepage recommendations for guest:")
    print(recommender.get_recommendations_for_homepage())