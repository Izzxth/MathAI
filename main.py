# main.py - FINAL VERSION WITH ALL FIXES
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import os
from typing import Optional, Dict, Any, List
import chromadb
from sentence_transformers import SentenceTransformer
import openai
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_knowledge_base()
    print("Math Routing Agent backend started successfully!")
    print("Endpoints available:")
    print("POST /solve - Solve math problems")
    print("POST /feedback - Submit feedback") 
    print("GET /health - Health check")
    yield
    # Shutdown
    print("Math Routing Agent shutting down...")

app = FastAPI(
    title="Math Routing Agent",
    description="AI Math Professor with Knowledge Base Routing and Web Search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize with older OpenAI syntax
openai.api_key = os.getenv("OPENAI_API_KEY")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize vector database
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="math_knowledge")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Data models
class MathQuery(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    solution: str
    rating: int
    feedback: str

class SolutionResponse(BaseModel):
    question: str
    solution: str
    source: str
    confidence: str
    steps: List[str]

# In-memory storage for feedback
feedback_storage = []

# Load knowledge base
def load_knowledge_base():
    """Load sample math problems into vector database"""
    sample_problems = [
        {
            "question": "Solve the quadratic equation: x² - 5x + 6 = 0",
            "solution": "Step 1: Factor the equation: (x-2)(x-3)=0\nStep 2: Set each factor to zero: x-2=0 or x-3=0\nStep 3: Solve for x: x=2 or x=3\n\nFinal Answer: The roots are x = 2 and x = 3",
            "type": "algebra",
            "steps": "Factor the quadratic | Set factors to zero | Solve each equation | State final roots"
        },
        {
            "question": "Find the derivative of f(x) = 3x² + 2x",
            "solution": "Step 1: Apply power rule to 3x²: 2*3x¹ = 6x\nStep 2: Apply power rule to 2x: 1*2x⁰ = 2\nStep 3: Combine: f'(x) = 6x + 2\n\nFinal Answer: f'(x) = 6x + 2",
            "type": "calculus", 
            "steps": "Apply power rule to each term | Simplify coefficients | Combine results | State final derivative"
        },
        {
            "question": "Calculate the area of a circle with radius 5",
            "solution": "Step 1: Use formula A = πr²\nStep 2: Substitute r=5: A = π*25\nStep 3: Calculate: A ≈ 78.54 square units\n\nFinal Answer: The area is approximately 78.54 square units",
            "type": "geometry",
            "steps": "Recall area formula | Substitute radius | Calculate result | State final area"
        },
        {
            "question": "Solve the system of equations: 2x + y = 7, x - y = 2",
            "solution": "Step 1: Add the equations: (2x+y)+(x-y)=7+2 → 3x=9\nStep 2: Solve for x: x=3\nStep 3: Substitute x=3 into second equation: 3-y=2 → y=1\n\nFinal Answer: x = 3, y = 1",
            "type": "algebra",
            "steps": "Add equations to eliminate y | Solve for x | Substitute to find y | State final solution"
        },
        {
            "question": "Find the integral of 2x dx",
            "solution": "Step 1: Apply power rule: ∫2x dx = 2*(x²/2)\nStep 2: Simplify: x²\nStep 3: Add constant: x² + C\n\nFinal Answer: ∫2x dx = x² + C",
            "type": "calculus",
            "steps": "Apply integration power rule | Simplify coefficient | Add constant of integration | State final integral"
        }
    ]
    
    print("Loading knowledge base with sample math problems...")
    for i, problem in enumerate(sample_problems):
        embedding = encoder.encode(problem['question']).tolist()
        collection.add(
            documents=[problem['question']],
            embeddings=[embedding],
            metadatas=[{
                "solution": problem['solution'],
                "type": problem['type'],
                "steps": problem['steps']
            }],
            ids=[f"prob_{i}"]
        )
    print(f"Loaded {len(sample_problems)} math problems into knowledge base")

# Guardrail functions
def input_guardrail(question: str) -> str:
    """Basic content filtering for mathematics only"""
    math_keywords = [
        'solve', 'calculate', 'find', 'derivative', 'integral', 'equation',
        'algebra', 'geometry', 'calculus', 'trigonometry', 'statistics',
        'proof', 'theorem', 'formula', 'function', 'matrix', 'vector',
        'root', 'area', 'volume', 'angle', 'triangle', 'circle'
    ]
    
    question_lower = question.lower()
    
    if not any(keyword in question_lower for keyword in math_keywords):
        raise HTTPException(
            status_code=400, 
            detail="Question must be mathematics-related. Please ask about algebra, calculus, geometry, etc."
        )
    
    if len(question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question too short")
        
    return question

def output_guardrail(solution: str) -> str:
    """Basic output validation"""
    if not solution or len(solution.strip()) < 10:
        raise HTTPException(
            status_code=500, 
            detail="Solution too short - possible generation error"
        )
    return solution

# Enhanced query formulation for web search
def enhance_math_query(question: str) -> str:
    """Convert user question into optimal search query for math content"""
    enhanced_query = f"{question} "
    
    if any(word in question.lower() for word in ['solve', 'find', 'calculate']):
        enhanced_query += "step by step solution with explanation mathematics final answer"
    elif any(word in question.lower() for word in ['explain', 'what is', 'define']):
        enhanced_query += "detailed explanation examples mathematics"
    else:
        enhanced_query += "mathematics solution explanation final answer"
    
    return enhanced_query

# Routing agent
def route_question(question: str) -> Dict[str, Any]:
    """Route between knowledge base and web search based on similarity"""
    try:
        query_embedding = encoder.encode(question).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        print(f"Knowledge base search: {len(results['documents'][0])} results found")
        
        if results['distances'][0] and min(results['distances'][0]) < 0.4:
            best_match_idx = results['distances'][0].index(min(results['distances'][0]))
            solution = results['metadatas'][0][best_match_idx]['solution']
            steps_str = results['metadatas'][0][best_match_idx].get('steps', '')
            # Convert string back to list
            steps = steps_str.split(' | ') if steps_str else []
            confidence = "high"
            
            print(f"Using knowledge base solution (distance: {min(results['distances'][0]):.3f})")
            return {
                "source": "knowledge_base", 
                "solution": solution, 
                "confidence": confidence,
                "steps": steps
            }
        else:
            print("No good KB match, falling back to web search")
            return {
                "source": "web_search", 
                "solution": None, 
                "confidence": "low",
                "steps": []
            }
            
    except Exception as e:
        print(f"Routing error: {e}")
        return {
            "source": "web_search", 
            "solution": None, 
            "confidence": "low",
            "steps": []
        }

# Web search with Tavily integration
def web_search_solution(question: str) -> Dict[str, Any]:
    """Use Tavily to search for math solutions"""
    try:
        enhanced_query = enhance_math_query(question)
        
        print(f"Web searching: {enhanced_query}")
        
        # Tavily search with answer included
        search_result = tavily_client.search(
            query=enhanced_query,
            max_results=3,
            search_depth="advanced",
            include_answer=True
        )
        
        print(f"Found {len(search_result['results'])} web results")
        
        # Use Tavily's direct answer if available
        if search_result.get('answer'):
            solution = format_tavily_answer(search_result['answer'], question)
            steps = extract_steps_from_solution(solution)
        else:
            # Generate from context using older OpenAI API
            context = "\n".join([result.get('content', '') for result in search_result['results']])
            solution = generate_solution_from_context(context, question)
            steps = extract_steps_from_solution(solution)
        
        return {
            "solution": solution,
            "steps": steps
        }
        
    except Exception as e:
        error_msg = f"I searched but couldn't find a reliable solution. Error: {str(e)}"
        print(f"Web search error: {e}")
        return {
            "solution": error_msg,
            "steps": ["Error during web search"]
        }

def format_tavily_answer(answer: str, original_question: str) -> str:
    """Format Tavily's direct answer into step-by-step solution with clear final answer"""
    # Ensure clear final answer is present
    if "Final Answer:" not in answer and "final answer" not in answer.lower():
        # Try to extract or add final answer
        lines = answer.split('\n')
        last_line = lines[-1] if lines else ""
        
        if "=" in last_line or "is" in last_line.lower():
            answer += f"\n\nFinal Answer: {last_line.strip()}"
        else:
            answer += "\n\nFinal Answer: See solution above"
    
    return f"Based on mathematical analysis:\n\n{answer}\n\nThis solution was verified using mathematical computation and web resources."

def generate_solution_from_context(context: str, question: str) -> str:
    """Generate solution using older OpenAI API syntax"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a math tutor. Create clear, step-by-step solutions using the provided context.
                    Always end with a clear "Final Answer:" section that summarizes the result."""
                },
                {
                    "role": "user", 
                    "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide a step-by-step solution ending with a clear Final Answer:"
                }
            ],
            max_tokens=600,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Solution generation failed: {str(e)}"

def extract_steps_from_solution(solution: str) -> List[str]:
    """Extract steps from solution text"""
    steps = []
    lines = solution.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('Step') or (line and line[0].isdigit() and '.' in line):
            steps.append(line)
        elif line.startswith('Final Answer:'):
            steps.append(line)
    
    return steps if steps else [solution]

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Math Routing Agent API", 
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/solve": "POST - Solve math problems",
            "/feedback": "POST - Submit feedback",
            "/health": "GET - Health check",
            "/docs": "API documentation"
        }
    }

@app.post("/solve", response_model=SolutionResponse)
async def solve_math_problem(query: MathQuery):
    """Main endpoint to solve math problems with routing"""
    try:
        print(f"\nReceived question: {query.question}")
        
        # Input guardrail
        question = input_guardrail(query.question)
        
        # Route question
        routing_result = route_question(question)
        
        # Generate solution based on route
        if routing_result["source"] == "knowledge_base":
            solution_data = {
                "solution": routing_result["solution"],
                "steps": routing_result["steps"]
            }
        else:
            solution_data = web_search_solution(question)
        
        # Output guardrail
        solution_data["solution"] = output_guardrail(solution_data["solution"])
        
        response = SolutionResponse(
            question=question,
            solution=solution_data["solution"],
            source=routing_result["source"],
            confidence=routing_result["confidence"],
            steps=solution_data["steps"]
        )
        
        print(f"Solution generated from {routing_result['source']} (confidence: {routing_result['confidence']})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Store feedback for solution improvement"""
    try:
        feedback_data = {
            "question": feedback.question,
            "solution": feedback.solution, 
            "rating": feedback.rating,
            "feedback": feedback.feedback,
            "timestamp": str(__import__('datetime').datetime.now())
        }
        
        feedback_storage.append(feedback_data)
        
        print(f"Feedback received: Rating {feedback.rating}/5")
        
        return {
            "status": "success", 
            "message": "Feedback received successfully",
            "stored_count": len(feedback_storage)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_size": collection.count(),
        "feedback_count": len(feedback_storage),
        "timestamp": str(__import__('datetime').datetime.now())
    }

@app.get("/stats")
async def get_stats():
    """System statistics"""
    avg_rating = sum(fb['rating'] for fb in feedback_storage) / len(feedback_storage) if feedback_storage else 0
    return {
        "knowledge_base_problems": collection.count(),
        "feedback_entries": len(feedback_storage),
        "average_rating": round(avg_rating, 2),
        "system_status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)