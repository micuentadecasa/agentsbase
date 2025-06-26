from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional

class OverallState(TypedDict):
    """State schema for E-commerce Inventory Management System.
    
    Implements hierarchical workflow with 3-tier architecture:
    - Tier 1: Data Collection (demand_forecasting, inventory_monitoring)
    - Tier 2: Analysis & Decision (supply_chain_intelligence, optimization_engine)  
    - Tier 3: Execution (procurement_automation)
    
    Follows TIP #010: Use Optional[List[...]] to prevent None + list concatenation errors.
    """
    
    # Shared Communication Channels
    messages: List[Dict[str, Any]]
    current_tier: str  # 'data_collection', 'analysis', 'execution'
    processing_status: str  # 'pending', 'in_progress', 'completed', 'error'
    
    # Tier 1 Data Collection Outputs
    demand_forecast: Optional[Dict[str, Any]]  # From demand_forecasting_agent
    inventory_status: Optional[Dict[str, Any]]  # From inventory_monitoring_agent
    
    # Tier 2 Analysis & Decision Outputs  
    supply_chain_analysis: Optional[Dict[str, Any]]  # From supply_chain_intelligence_agent
    optimization_strategy: Optional[Dict[str, Any]]   # From optimization_engine_agent
    
    # Tier 3 Execution Outputs
    procurement_actions: Optional[List[Dict[str, Any]]]  # From procurement_automation_agent
    
    # Cross-cutting Intelligence Data
    market_intelligence: Optional[Dict[str, Any]]  # Market trends, competitor data
    performance_metrics: Optional[Dict[str, Any]]   # KPIs, success metrics
    
    # Error Handling & Communication (TIP #010 compliance)
    errors: Optional[List[Dict[str, Any]]]                # Error tracking with None safety
    agent_communications: Optional[List[Dict[str, Any]]]  # Inter-agent messaging
    
    # Workflow Metadata
    workflow_id: str
    timestamp: str
    business_context: Optional[Dict[str, Any]]  # E-commerce specific context
    
    # Hierarchical Flow Control
    tier_1_complete: Optional[bool]  # Data collection finished
    tier_2_complete: Optional[bool]  # Analysis finished  
    tier_3_complete: Optional[bool]  # Execution finished
    
    # Emergency Handling Flags
    stockout_detected: Optional[bool]      # Emergency stockout situation
    critical_supply_risk: Optional[bool]   # Critical supply chain disruption 