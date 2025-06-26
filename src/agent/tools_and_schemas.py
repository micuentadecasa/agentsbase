from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3
import requests
from bs4 import BeautifulSoup
import json
import random
import time

# =====================================
# PYDANTIC SCHEMAS FOR DATA VALIDATION
# =====================================

class SKUData(BaseModel):
    """Schema for product SKU information"""
    sku_id: str = Field(..., description="Unique SKU identifier")
    product_name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    current_stock: int = Field(..., description="Current inventory level")
    unit_cost: float = Field(..., description="Unit cost in USD")
    selling_price: float = Field(..., description="Current selling price")

class DemandForecast(BaseModel):
    """Schema for demand forecasting results"""
    sku_id: str = Field(..., description="SKU identifier")
    predicted_demand: int = Field(..., description="Predicted demand quantity")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    seasonality_factor: float = Field(..., description="Seasonal adjustment factor")
    market_trend: str = Field(..., description="Market trend analysis")

class InventoryStatus(BaseModel):
    """Schema for inventory monitoring results"""
    sku_id: str = Field(..., description="SKU identifier")
    current_stock: int = Field(..., description="Current stock level")
    warehouse_location: str = Field(..., description="Primary warehouse location")
    last_updated: str = Field(..., description="Last inventory update timestamp")
    stock_condition: str = Field(..., description="Stock condition assessment")

class SupplierPerformance(BaseModel):
    """Schema for supplier performance metrics"""
    supplier_id: str = Field(..., description="Unique supplier identifier")
    supplier_name: str = Field(..., description="Supplier company name")
    lead_time_days: int = Field(..., description="Average lead time in days")
    reliability_score: float = Field(..., ge=0, le=1, description="Supplier reliability (0-1)")
    risk_level: str = Field(..., description="Risk assessment level")

class OptimizationRecommendation(BaseModel):
    """Schema for optimization engine recommendations"""
    sku_id: str = Field(..., description="SKU identifier")
    reorder_point: int = Field(..., description="Recommended reorder point")
    safety_stock: int = Field(..., description="Recommended safety stock level")
    order_quantity: int = Field(..., description="Recommended order quantity")
    priority_level: str = Field(..., description="Urgency priority level")

class PurchaseOrder(BaseModel):
    """Schema for purchase order generation"""
    order_id: str = Field(..., description="Unique order identifier")
    supplier_id: str = Field(..., description="Target supplier")
    sku_id: str = Field(..., description="Product SKU")
    quantity: int = Field(..., description="Order quantity")
    unit_price: float = Field(..., description="Negotiated unit price")
    total_amount: float = Field(..., description="Total order amount")
    expected_delivery: str = Field(..., description="Expected delivery date")

# =====================================
# DATABASE INTEGRATION TOOLS
# =====================================

def initialize_sample_database():
    """Initialize SQLite database with sample e-commerce data"""
    conn = sqlite3.connect(':memory:')  # In-memory database for demo
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE sales_history (
            id INTEGER PRIMARY KEY,
            sku_id TEXT,
            sale_date TEXT,
            quantity_sold INTEGER,
            unit_price REAL,
            customer_segment TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE inventory (
            sku_id TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            current_stock INTEGER,
            warehouse_location TEXT,
            unit_cost REAL,
            last_updated TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE suppliers (
            supplier_id TEXT PRIMARY KEY,
            supplier_name TEXT,
            contact_email TEXT,
            lead_time_days INTEGER,
            reliability_score REAL
        )
    ''')
    
    # Insert sample data
    sample_sales = [
        ('SKU001', '2024-01-15', 50, 29.99, 'retail'),
        ('SKU001', '2024-01-16', 35, 29.99, 'wholesale'),
        ('SKU002', '2024-01-15', 25, 15.99, 'retail'),
        ('SKU003', '2024-01-17', 100, 8.99, 'online'),
    ]
    
    sample_inventory = [
        ('SKU001', 'Wireless Headphones', 'Electronics', 150, 'WH001', 18.50, '2024-01-26'),
        ('SKU002', 'Phone Case', 'Accessories', 300, 'WH001', 8.99, '2024-01-26'),
        ('SKU003', 'USB Cable', 'Electronics', 500, 'WH002', 4.25, '2024-01-26'),
    ]
    
    sample_suppliers = [
        ('SUP001', 'TechCorp Supply', 'orders@techcorp.com', 7, 0.95),
        ('SUP002', 'Global Electronics', 'supply@globalelec.com', 14, 0.88),
        ('SUP003', 'FastShip Components', 'orders@fastship.com', 3, 0.92),
    ]
    
    cursor.executemany('INSERT INTO sales_history (sku_id, sale_date, quantity_sold, unit_price, customer_segment) VALUES (?, ?, ?, ?, ?)', sample_sales)
    cursor.executemany('INSERT INTO inventory (sku_id, product_name, category, current_stock, warehouse_location, unit_cost, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)', sample_inventory)
    cursor.executemany('INSERT INTO suppliers (supplier_id, supplier_name, contact_email, lead_time_days, reliability_score) VALUES (?, ?, ?, ?, ?)', sample_suppliers)
    
    conn.commit()
    return conn

def database_query_tool(query: str, connection=None) -> List[Dict[str, Any]]:
    """Execute database queries for sales and inventory data"""
    if connection is None:
        connection = initialize_sample_database()
    
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
    except Exception as e:
        return [{"error": f"Database query failed: {str(e)}"}]

# =====================================
# WEB SCRAPING TOOLS  
# =====================================

def web_scraping_tool(target_url: str, scrape_type: str = "pricing") -> Dict[str, Any]:
    """Web scraping tool for competitive intelligence and market data
    
    Args:
        target_url: URL to scrape (for demo, we'll simulate)
        scrape_type: Type of data to scrape ('pricing', 'trends', 'news')
    """
    # Simulate rate limiting
    time.sleep(random.uniform(0.5, 2.0))
    
    try:
        # For demo purposes, return simulated competitive data
        if scrape_type == "pricing":
            return {
                "competitor_prices": {
                    "SKU001": {"competitor_a": 27.99, "competitor_b": 31.99, "avg_market_price": 29.99},
                    "SKU002": {"competitor_a": 14.99, "competitor_b": 17.99, "avg_market_price": 16.49},
                    "SKU003": {"competitor_a": 7.99, "competitor_b": 9.99, "avg_market_price": 8.99}
                },
                "scrape_timestamp": datetime.now().isoformat(),
                "source_reliability": 0.85
            }
        elif scrape_type == "trends":
            return {
                "market_trends": [
                    {"category": "Electronics", "trend": "increasing", "growth_rate": 0.15},
                    {"category": "Accessories", "trend": "stable", "growth_rate": 0.03},
                ],
                "seasonal_factors": {
                    "electronics": 1.2,  # 20% seasonal boost
                    "accessories": 1.05   # 5% seasonal boost
                },
                "scrape_timestamp": datetime.now().isoformat()
            }
        else:  # news/supply chain
            return {
                "supply_chain_alerts": [
                    {"severity": "low", "message": "Minor delays expected in Asia shipping routes"},
                    {"severity": "medium", "message": "Electronics component shortage in Q2 predicted"}
                ],
                "market_news": [
                    {"title": "E-commerce demand continues growth trend", "impact": "positive"},
                    {"title": "New trade regulations may affect electronics imports", "impact": "negative"}
                ],
                "scrape_timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {
            "error": f"Web scraping failed: {str(e)}",
            "fallback_data": "Using cached competitive intelligence"
        }

# =====================================
# API INTEGRATION TOOLS
# =====================================

def supplier_api_tool(supplier_id: str, action: str = "get_info") -> Dict[str, Any]:
    """Integration with supplier APIs for performance metrics and orders
    
    Args:
        supplier_id: Unique supplier identifier
        action: API action ('get_info', 'get_performance', 'place_order')
    """
    try:
        # Simulate API call delay
        time.sleep(random.uniform(0.3, 1.0))
        
        supplier_data = {
            "SUP001": {
                "name": "TechCorp Supply",
                "performance": {"on_time_delivery": 0.95, "quality_score": 0.92, "lead_time": 7},
                "current_capacity": "high",
                "pricing_tier": "preferred"
            },
            "SUP002": {
                "name": "Global Electronics", 
                "performance": {"on_time_delivery": 0.88, "quality_score": 0.90, "lead_time": 14},
                "current_capacity": "medium",
                "pricing_tier": "standard"
            },
            "SUP003": {
                "name": "FastShip Components",
                "performance": {"on_time_delivery": 0.92, "quality_score": 0.89, "lead_time": 3},
                "current_capacity": "low",
                "pricing_tier": "premium"
            }
        }
        
        if action == "get_info":
            return supplier_data.get(supplier_id, {"error": "Supplier not found"})
        elif action == "get_performance":
            return supplier_data.get(supplier_id, {}).get("performance", {"error": "Performance data unavailable"})
        elif action == "place_order":
            return {
                "order_status": "pending",
                "order_id": f"ORD{random.randint(10000, 99999)}",
                "estimated_delivery": "7-14 days",
                "confirmation": True
            }
            
    except Exception as e:
        return {"error": f"Supplier API call failed: {str(e)}"}

def inventory_api_tool(sku_id: str = None, action: str = "get_status") -> Dict[str, Any]:
    """Integration with inventory management system APIs
    
    Args:
        sku_id: Product SKU (if None, returns all inventory)
        action: API action ('get_status', 'update_stock', 'check_capacity')
    """
    try:
        # Simulate real API delay
        time.sleep(random.uniform(0.2, 0.8))
        
        inventory_data = {
            "SKU001": {"current_stock": 150, "location": "WH001", "condition": "good", "last_movement": "2024-01-26"},
            "SKU002": {"current_stock": 300, "location": "WH001", "condition": "excellent", "last_movement": "2024-01-25"},
            "SKU003": {"current_stock": 500, "location": "WH002", "condition": "good", "last_movement": "2024-01-24"}
        }
        
        if action == "get_status":
            if sku_id:
                return inventory_data.get(sku_id, {"error": "SKU not found"})
            else:
                return {"all_inventory": inventory_data, "total_skus": len(inventory_data)}
        elif action == "check_capacity":
            return {
                "WH001": {"total_capacity": 10000, "current_usage": 4500, "available": 5500},
                "WH002": {"total_capacity": 15000, "current_usage": 8200, "available": 6800}
            }
            
    except Exception as e:
        return {"error": f"Inventory API call failed: {str(e)}"}

def market_intelligence_tool(data_type: str = "trends") -> Dict[str, Any]:
    """Market intelligence and trend analysis tool
    
    Args:
        data_type: Type of market data ('trends', 'competitors', 'demand_signals')
    """
    try:
        # Simulate market data API call
        time.sleep(random.uniform(0.5, 1.5))
        
        if data_type == "trends":
            return {
                "category_trends": {
                    "electronics": {"growth_rate": 0.15, "trend_direction": "up", "confidence": 0.88},
                    "accessories": {"growth_rate": 0.03, "trend_direction": "stable", "confidence": 0.75}
                },
                "seasonal_indicators": {
                    "current_season": "winter",
                    "next_season_impact": "spring_electronics_boost",
                    "seasonal_multiplier": 1.2
                }
            }
        elif data_type == "competitors":
            return {
                "competitor_analysis": {
                    "price_positioning": "competitive",
                    "market_share_trend": "growing",
                    "competitive_pressure": "medium"
                },
                "competitive_alerts": [
                    {"competitor": "CompetitorA", "action": "price_reduction", "impact": "medium"},
                    {"competitor": "CompetitorB", "action": "new_product_launch", "impact": "low"}
                ]
            }
        else:  # demand_signals
            return {
                "demand_signals": {
                    "social_sentiment": "positive",
                    "search_volume_trend": "increasing",
                    "customer_engagement": "high"
                },
                "predictive_indicators": {
                    "demand_acceleration": 0.12,
                    "market_saturation_risk": "low",
                    "emerging_categories": ["wireless_audio", "mobile_accessories"]
                }
            }
            
    except Exception as e:
        return {"error": f"Market intelligence API failed: {str(e)}"}

# =====================================
# OPTIMIZATION & CALCULATION TOOLS
# =====================================

def optimization_algorithm_tool(inventory_data: Dict[str, Any], demand_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mathematical optimization for inventory parameters
    
    Args:
        inventory_data: Current inventory status
        demand_data: Demand forecasting results
    """
    try:
        optimization_results = {}
        
        for sku_id, demand_info in demand_data.items():
            if isinstance(demand_info, dict) and "predicted_demand" in demand_info:
                predicted_demand = demand_info["predicted_demand"]
                confidence = demand_info.get("confidence_score", 0.8)
                
                # Simple EOQ-based optimization (Economic Order Quantity)
                carrying_cost_rate = 0.25  # 25% annual carrying cost
                ordering_cost = 50  # $50 per order
                annual_demand = predicted_demand * 52  # Weekly to annual
                
                if annual_demand > 0:
                    eoq = int((2 * annual_demand * ordering_cost / carrying_cost_rate) ** 0.5)
                    safety_stock = int(predicted_demand * 0.5 * (1 + confidence))  # Safety factor based on confidence
                    reorder_point = int(predicted_demand * 2) + safety_stock  # 2-week lead time assumption
                    
                    optimization_results[sku_id] = {
                        "economic_order_quantity": eoq,
                        "reorder_point": reorder_point,
                        "safety_stock": safety_stock,
                        "recommended_action": "reorder" if inventory_data.get(sku_id, {}).get("current_stock", 0) <= reorder_point else "monitor"
                    }
        
        return {
            "optimization_results": optimization_results,
            "algorithm_used": "EOQ_with_safety_stock",
            "optimization_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Optimization calculation failed: {str(e)}"}

def purchase_order_tool(sku_id: str, quantity: int, supplier_id: str) -> Dict[str, Any]:
    """Generate and manage purchase orders
    
    Args:
        sku_id: Product SKU to order
        quantity: Order quantity
        supplier_id: Target supplier
    """
    try:
        # Simulate purchase order generation
        order_id = f"PO{random.randint(100000, 999999)}"
        
        # Get supplier pricing (simulated)
        pricing_data = {
            "SUP001": {"SKU001": 18.50, "SKU002": 8.99, "SKU003": 4.25},
            "SUP002": {"SKU001": 19.00, "SKU002": 9.25, "SKU003": 4.50},
            "SUP003": {"SKU001": 17.75, "SKU002": 8.75, "SKU003": 4.00}
        }
        
        unit_price = pricing_data.get(supplier_id, {}).get(sku_id, 0.0)
        total_amount = unit_price * quantity
        
        return {
            "purchase_order": {
                "order_id": order_id,
                "sku_id": sku_id,
                "supplier_id": supplier_id,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": total_amount,
                "order_date": datetime.now().isoformat(),
                "expected_delivery": "7-14 days",
                "status": "pending_approval"
            },
            "supplier_notification": {
                "sent": True,
                "method": "API",
                "confirmation_expected": "24 hours"
            }
        }
        
    except Exception as e:
        return {"error": f"Purchase order generation failed: {str(e)}"}

# =====================================
# SAMPLE DATA GENERATORS
# =====================================

def create_sample_demand_forecast() -> Dict[str, Any]:
    """Generate sample demand forecast data for testing"""
    return {
        "SKU001": {
            "predicted_demand": 45,
            "confidence_score": 0.85,
            "seasonality_factor": 1.2,
            "market_trend": "increasing"
        },
        "SKU002": {
            "predicted_demand": 30,
            "confidence_score": 0.78,
            "seasonality_factor": 1.05,
            "market_trend": "stable"
        },
        "SKU003": {
            "predicted_demand": 85,
            "confidence_score": 0.92,
            "seasonality_factor": 1.1,
            "market_trend": "strong_growth"
        }
    }

def create_sample_inventory_status() -> Dict[str, Any]:
    """Generate sample inventory status data for testing"""
    return {
        "SKU001": {
            "current_stock": 150,
            "warehouse_location": "WH001", 
            "last_updated": "2024-01-26T10:30:00Z",
            "stock_condition": "good"
        },
        "SKU002": {
            "current_stock": 300,
            "warehouse_location": "WH001",
            "last_updated": "2024-01-26T10:30:00Z", 
            "stock_condition": "excellent"
        },
        "SKU003": {
            "current_stock": 500,
            "warehouse_location": "WH002",
            "last_updated": "2024-01-26T10:30:00Z",
            "stock_condition": "good"
        }
    } 