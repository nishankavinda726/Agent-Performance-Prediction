import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AgentRecommendationSystem:
    """
    A system to analyze agent performance and provide personalized recommendations
    based on key performance indicators.
    """
    
    def __init__(self):
        # Define optimal thresholds based on feature importance analysis
        self.thresholds = {
            'customer_to_policy_ratio': 5.0,  # Lower is better, threshold where performance drops
            'unique_proposal': 15,  # Higher is better (minimum target)
            'proposal_to_policy_ratio': 5.0,  # Lower is better
            'quotation_to_policy_ratio': 5.0,  # Lower is better
            'cash_payment_policies': 10,  # Higher is better (target)
            'cash_policy_to_proposal_ratio': 0.5  # Higher is better (target)
        }
        
        # Define recommendation actions for each performance area
        self.action_plans = {
            'customer_to_policy_ratio': [
                "Focus on quality customer interactions rather than increasing customer volume",
                "Implement follow-up system for existing customers",
                "Review sales scripts to improve policy conversion discussions",
                "Analyze which customer segments have higher conversion rates and prioritize them"
            ],
            'unique_proposal': [
                "Increase proposal diversity by tailoring to different customer needs",
                "Learn about full product range to offer more appropriate options",
                "Create proposal templates for different customer segments",
                "Study top performers' proposal strategies and adapt them"
            ],
            'proposal_to_policy_ratio': [
                "Improve proposal quality rather than quantity",
                "Focus on better understanding customer needs before proposing policies",
                "Implement structured follow-up process after proposal submission",
                "Role-play objection handling to improve conversion skills"
            ],
            'quotation_to_policy_ratio': [
                "Review quotation process to identify conversion barriers",
                "Improve follow-up timing after quotation delivery",
                "Address common customer concerns pre-emptively in quotations",
                "Implement comparison tools to demonstrate value against competitors"
            ],
            'cash_payment_policies': [
                "Highlight benefits of cash payment options to customers",
                "Create special promotions for cash payment policies",
                "Learn to effectively explain cash payment advantages",
                "Target customer segments more likely to choose cash payment"
            ],
            'cash_policy_to_proposal_ratio': [
                "Incorporate cash payment benefits earlier in sales discussions",
                "Develop better incentives for cash payment selection",
                "Create clear comparison materials showing cash vs. other payment benefits",
                "Study successful agents' techniques for promoting cash policies"
            ]
        }
        
        # Define prioritization weights for recommendations
        self.priority_weights = {
            'customer_to_policy_ratio': 5,  # Highest importance
            'unique_proposal': 4,
            'proposal_to_policy_ratio': 3,
            'quotation_to_policy_ratio': 3,
            'cash_payment_policies': 2,
            'cash_policy_to_proposal_ratio': 2
        }
    
    def analyze_agent(self, agent_metrics):
        """
        Analyze an agent's metrics and identify areas for improvement
        
        Parameters:
        -----------
        agent_metrics: dict
            Dictionary containing agent performance metrics
            
        Returns:
        --------
        dict: Performance gaps and recommendations
        """
        gaps = {}
        
        # Identify performance gaps
        for metric, threshold in self.thresholds.items():
            if metric not in agent_metrics:
                continue
                
            value = agent_metrics[metric]
            
            # For metrics where lower is better
            if metric in ['customer_to_policy_ratio', 'proposal_to_policy_ratio', 'quotation_to_policy_ratio']:
                if value > threshold:
                    gap_size = (value - threshold) / threshold  # Relative gap size
                    gaps[metric] = {
                        'current': value,
                        'target': threshold,
                        'gap_size': gap_size,
                        'priority': gap_size * self.priority_weights[metric]
                    }
            # For metrics where higher is better
            else:
                if value < threshold:
                    gap_size = (threshold - value) / threshold  # Relative gap size
                    gaps[metric] = {
                        'current': value,
                        'target': threshold,
                        'gap_size': gap_size,
                        'priority': gap_size * self.priority_weights[metric]
                    }
        
        return gaps
    
    def generate_recommendations(self, gaps, max_recommendations=3):
        """
        Generate personalized recommendations based on performance gaps
        
        Parameters:
        -----------
        gaps: dict
            Dictionary of performance gaps
        max_recommendations: int
            Maximum number of recommendation areas to include
            
        Returns:
        --------
        list: Prioritized recommendations
        """
        if not gaps:
            return ["Performance metrics are within optimal ranges. Focus on maintaining consistent results."]
        
        # Sort gaps by priority (highest first)
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1]['priority'], reverse=True)
        
        # Limit to max_recommendations
        top_gaps = sorted_gaps[:max_recommendations]
        
        recommendations = []
        for metric, gap_info in top_gaps:
            rec = {
                'area': metric,
                'current_value': gap_info['current'],
                'target_value': gap_info['target'],
                'priority': gap_info['priority'],
                'actions': self.action_plans[metric]
            }
            recommendations.append(rec)
            
        return recommendations
    
    def generate_action_plan(self, agent_metrics):
        """
        Create a complete action plan for an agent
        
        Parameters:
        -----------
        agent_metrics: dict
            Dictionary containing agent performance metrics
            
        Returns:
        --------
        dict: Complete personalized action plan
        """
        gaps = self.analyze_agent(agent_metrics)
        recommendations = self.generate_recommendations(gaps)
        
        action_plan = {
            'agent_metrics': agent_metrics,
            'performance_gaps': gaps,
            'recommendations': recommendations
        }
        
        return action_plan
    
    def format_action_plan(self, action_plan):
        """
        Format the action plan as a readable report
        
        Parameters:
        -----------
        action_plan: dict
            Complete action plan
            
        Returns:
        --------
        str: Formatted action plan
        """
        recommendations = action_plan['recommendations']
        
        if not recommendations:
            return "Performance is within optimal ranges across all key metrics. Keep up the good work!"
        
        formatted_plan = "# PERSONALIZED AGENT ACTION PLAN\n\n"
        formatted_plan += "## Performance Summary\n\n"
        
        # Add metrics summary
        metrics = action_plan['agent_metrics']
        formatted_plan += "Current metrics:\n"
        for metric, value in metrics.items():
            if metric in self.thresholds:
                formatted_plan += f"- {metric}: {value:.2f}"
                if metric in action_plan['performance_gaps']:
                    formatted_plan += f" (Target: {self.thresholds[metric]:.2f}) ⚠️\n"
                else:
                    formatted_plan += f" (Target: {self.thresholds[metric]:.2f}) ✅\n"
        
        formatted_plan += "\n## Prioritized Action Items\n\n"
        
        # Add recommendations
        for i, rec in enumerate(recommendations, 1):
            formatted_plan += f"### Priority {i}: Improve {rec['area'].replace('_', ' ').title()}\n\n"
            formatted_plan += f"Current value: {rec['current_value']:.2f} | Target: {rec['target_value']:.2f}\n\n"
            formatted_plan += "Recommended actions:\n"
            for j, action in enumerate(rec['actions'], 1):
                formatted_plan += f"{j}. {action}\n"
            formatted_plan += "\n"
        
        return formatted_plan

    def visualize_performance(self, agent_metrics):
        """
        Create visualizations of agent performance compared to optimal thresholds
        
        Parameters:
        -----------
        agent_metrics: dict
            Dictionary containing agent performance metrics
            
        Returns:
        --------
        matplotlib.figure: Performance visualization
        """
        metrics = []
        values = []
        targets = []
        colors = []
        
        for metric, value in agent_metrics.items():
            if metric in self.thresholds:
                metrics.append(metric.replace('_', ' ').title())
                values.append(value)
                targets.append(self.thresholds[metric])
                
                # Determine if meeting target (different logic for different metrics)
                if metric in ['customer_to_policy_ratio', 'proposal_to_policy_ratio', 'quotation_to_policy_ratio']:
                    # Lower is better
                    colors.append('green' if value <= self.thresholds[metric] else 'red')
                else:
                    # Higher is better
                    colors.append('green' if value >= self.thresholds[metric] else 'red')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set positions for bars
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, values, width, label='Current', color=colors)
        ax.bar(x + width/2, targets, width, label='Target', color='lightgray')
        
        # Add labels and title
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Agent Performance vs Targets')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize system
    rec_system = AgentRecommendationSystem()
    
    # Example agent data
    agent1 = {
        'customer_to_policy_ratio': 7.5,  # Above threshold (bad)
        'unique_proposal': 8,  # Below threshold (bad)
        'proposal_to_policy_ratio': 4.2,  # Below threshold (good)
        'quotation_to_policy_ratio': 6.1,  # Above threshold (bad)
        'cash_payment_policies': 7,  # Below threshold (bad)
        'cash_policy_to_proposal_ratio': 0.3  # Below threshold (bad)
    }
    
    # Generate action plan
    action_plan = rec_system.generate_action_plan(agent1)
    
    # Print formatted plan
    print(rec_system.format_action_plan(action_plan))
    
    # Create visualization
    fig = rec_system.visualize_performance(agent1)
    plt.show()
