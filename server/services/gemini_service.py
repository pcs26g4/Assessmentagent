import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from .determinism_config import DeterministicEvalConfig, EvaluationCache

load_dotenv()

logger = logging.getLogger(__name__)

# Constants for retry logic
MAX_LLM_RETRIES = 3
BACKOFF_BASE = 1.0

# Validate determinism configuration on import
DeterministicEvalConfig.validate_configuration()

class ExtractedQA(BaseModel):
    question: str = Field(description="The question text extracted from the document")
    student_answer: str = Field(description="The student's answer text extracted from the document")
    is_answer_present: bool = Field(description="Whether an answer was found for this question")

class ExtractedQAList(BaseModel):
    qa_pairs: List[ExtractedQA] = Field(description="List of question-answer pairs extracted from the document")

class EvalDetail(BaseModel):
    question: str = Field(description="The question being evaluated")
    student_answer: str = Field(description="The student's answer being evaluated")
    correct_answer: str = Field(description="The correct answer for the question")
    is_correct: bool = Field(description="Whether the student's answer is correct")
    partial_credit: Optional[float] = Field(None, description="Partial credit score (0.0, 0.25, 0.5, 0.75, 1.0)", ge=0.0, le=1.0)
    max_marks: float = Field(1.0, description="The maximum marks for this question found in the text (e.g., 5 or 10). Default 1.0.")
    feedback: str = Field(description="Detailed feedback on the student's answer")

class PPTEvalCriteria(BaseModel):
    score: int = Field(description="Score between 0 and 100")
    feedback: str = Field(description="Brief feedback on the criteria")

class PPTEvaluation(BaseModel):
    content_quality: PPTEvalCriteria
    structure: PPTEvalCriteria
    alignment: PPTEvalCriteria
    strengths: List[str]
    improvements: List[str]
    summary: str

class PPTDesignEvaluation(BaseModel):
    visual_clarity: PPTEvalCriteria
    layout_balance: PPTEvalCriteria
    color_consistency: PPTEvalCriteria
    typography: PPTEvalCriteria
    visual_appeal: PPTEvalCriteria
    design_strengths: List[str]
    design_improvements: List[str]
    design_summary: str

class GitProjectInfo(BaseModel):
    project_about: str
    project_use: str
    technology_stack: List[str]
    features: List[str]
    project_structure: str

class GitRuleResult(BaseModel):
    rule_text: str
    is_satisfied: bool
    severity: str
    evidence: str
    failure_reason: str

class GitTechMismatch(BaseModel):
    expected_from_description: str
    actual_from_code: str
    has_mismatch: bool
    details: str

class GitGradingResult(BaseModel):
    rules_summary: str
    overall_comment: str
    conversational_response: str = Field(description="Direct, conversational answer to the user's specific question/description about the code.")
    score_percent: Optional[float] = Field(0.0, description="Deprecated/Filtered out score")
    detected_technology_stack: List[str]
    rule_results: List[GitRuleResult]
    technology_mismatch: GitTechMismatch

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        # CRITICAL: Force fixed model for determinism
        self.model = DeterministicEvalConfig.FIXED_MODEL
        logger.info(f"🔐 Using FIXED model for determinism: {self.model}")
        self.max_retries = MAX_LLM_RETRIES
        self.backoff_base = BACKOFF_BASE
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY not found in environment")

    def _get_client(self):
        if not self.client:
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
        return self.client

    async def _call_gemini_core(self, contents: Any, config: types.GenerateContentConfig, response_schema: Optional[Any] = None, operation_name: str = "LLM Call") -> Dict:
        """
        Robust core wrapper for Gemini SDK with exponential retry and standardized error handling.
        """
        client = self._get_client()
        if not client:
            return {
                "success": False, 
                "error": {
                    "type": "CONFIG_ERROR", 
                    "message": "Gemini API key is missing",
                    "status_code": None,
                    "raw": "Client not initialized"
                }
            }

        attempt = 0
        last_error_msg = ""
        last_status_code = None

        while attempt <= self.max_retries:
            try:
                # DEBUG: Log the model being used
                print(f"\n🤖 [ACTIVE MODEL] Operation: '{operation_name}' is using Model: '{self.model}'")
                print("-" * 50)

                # DEBUG: Print exact input being sent to LLM
                print(f"🚀 [LLM INPUT] {operation_name} (Attempt {attempt+1}):")
                print("-" * 50)
                # Handle both string prompts and part-based prompts (vision)
                if isinstance(contents, str):
                    print(contents)
                elif isinstance(contents, list):
                    for part in contents:
                        if isinstance(part, str): print(part)
                        else: print(f"[Binary Part: {type(part)}]")
                print("-" * 50)

                # Use thread pool for blocking SDK call
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    )
                )

                # Successful execution
                raw_text = response.text or ""
                
                # DEBUG: Print exact output received from LLM
                print(f"\n✨ [LLM OUTPUT] {operation_name}:")
                print("-" * 50)
                print(raw_text)
                print("-" * 50 + "\n")

                if response_schema:
                    try:
                        data = response_schema.model_validate_json(raw_text)
                        return {"success": True, "response": data}
                    except Exception as parse_err:
                        logger.error(f"Structured parse failed for {operation_name}: {parse_err}")
                        return {
                            "success": False,
                            "error": {
                                "type": "PARSE_ERROR",
                                "message": f"Failed to parse structured output from {operation_name}",
                                "status_code": 200,
                                "raw": str(parse_err)
                            }
                        }
                else:
                    return {"success": True, "response": raw_text}

            except Exception as e:
                last_error_msg = str(e)
                last_status_code = getattr(e, 'status_code', None)
                
                # Detect status codes from message if not provided
                if last_status_code is None:
                    for code in [429, 500, 502, 503, 504]:
                        if str(code) in last_error_msg:
                            last_status_code = code
                            break
                
                retryable_codes = {429, 500, 502, 503, 504}
                retry_strings = ["overloaded", "timeout", "deadline", "connection", "rate limit", "busy"]
                
                is_retryable = (last_status_code in retryable_codes) or \
                               any(s in last_error_msg.lower() for s in retry_strings)
                
                if is_retryable and attempt < self.max_retries:
                    delay = 2 ** attempt
                    logger.warning(f"{operation_name} failed (status={last_status_code}), retrying in {delay}s... (Attempt {attempt+1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                
                # Final failure
                logger.error(f"{operation_name} failed permanently after {attempt+1} attempts: {last_error_msg}")
                return {
                    "success": False,
                    "error": {
                        "type": "LLM_UNAVAILABLE",
                        "message": "LLM service unavailable after retries (e.g., 503 Model overloaded). Please try again later.",
                        "status_code": last_status_code,
                        "raw": last_error_msg
                    }
                }

        return {
            "success": False,
            "error": {
                "type": "LLM_UNAVAILABLE",
                "message": "LLM service exceeded maximum retry attempts.",
                "status_code": last_status_code,
                "raw": last_error_msg
            }
        }

    async def extract_qa_structured(self, text: str) -> Dict:
        """
        Uses Gemini structured output to extract QA pairs with standardized return.
        DETERMINISTIC: Uses content hashing and caching to ensure same results.
        """
        # Generate content hash for caching
        content_hash = DeterministicEvalConfig.get_content_hash(text)
        
        # Check cache first
        cached_result = EvaluationCache.get(content_hash, eval_type="qa_extraction")
        if cached_result is not None:
            return cached_result
        
        # Standardized, deterministic prompt
        prompt = f"""### ROLE: You are a senior backend engineer and NLP specialist.
Analyze and extract Question–Answer pairs from the provided text.

### EXTRACTION RULES (STRICT & DETERMINISTIC):
1. **EXPLICIT STRUCTURE ONLY**: You must ONLY extract pairs if there is a clear "Question" and "Answer" format (e.g., "Q1:", "Question:", "Q.", "1.").
2. **NO GUESSING**: If the text contains ONLY answers (e.g., "The code implements a sorting algorithm..."), DO NOT guess or infer what the question might have been. Return an empty list.
3. **Question patterns**: Q, Q., Q:, Q), Question, Ques, Qus (case-insensitive)
4. **Answer patterns**: Answer, Ans, A, A:, A., A) (case-insensitive)
5. **Extract FULL multi-line text** for each field.
6. **Preserve original formatting** and content exactly.
7. **FAILURE CONDITION**: If no clear Q&A structure is found, return an empty list "[]".

### TEXT TO ANALYZE:
{text}

### OUTPUT REQUIREMENTS:
- Return ONLY valid JSON matching the schema
- Each qa_pair MUST have: question (string), student_answer (string), is_answer_present (boolean)
- Never skip or summarize content"""
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=ExtractedQAList.model_json_schema(),
        )
        
        res = await self._call_gemini_core(prompt, config, ExtractedQAList, "QA Extraction")
        
        if res["success"]:
            # Flatten to compatibility format
            res["response"] = [qa.model_dump() for qa in res["response"].qa_pairs]
            # Cache successful result
            EvaluationCache.set(content_hash, res, eval_type="qa_extraction")
        
        return res

    async def evaluate_one_qa(self, description: str, question: str, student_answer: str, question_index: int = 1) -> Dict:
        """
        Standardized per-question evaluation using strict atomic call with structured output.
        DETERMINISTIC: Uses content hashing, caching, and consensus voting.
        """
        # Create deterministic content hash for caching
        combined_input = f"{description}|||{question}|||{student_answer}|||{question_index}"
        content_hash = DeterministicEvalConfig.get_content_hash(combined_input)
        
        # Check cache first
        cached_result = EvaluationCache.get(content_hash, eval_type="qa_evaluation")
        if cached_result is not None:
            return cached_result
        
        # Standardized, deterministic prompt with exact scoring rules
        prompt = f"""### ROLE: You are a strict and consistent academic grader.
Evaluate the student's answer based ONLY on the provided rubric and question.

### ASSIGNMENT DESCRIPTION/RUBRIC:
{description}

### QUESTION NUMBER: {question_index}
### QUESTION:
{question}

### STUDENT ANSWER:
{student_answer}

### GRADING PROCESS (STEP-BY-STEP):
1. **Analyze Requirements**: Check if the description assigns specific marks/points to this question (e.g., "5 marks", "10 points").
   - If YES: Set `max_marks` to this value (e.g., 5.0). Grade strictly out of those points. Convert the result to a 0.0-1.0 scale (e.g., 3/5 -> 0.6).
   - If NO: Set `max_marks` to 1.0. Use the standard default impact criteria below.

2. **Verify Correctness**:
   - Compare the Student Answer against the Question requirements.
   - Check for: Exactness, Logic, Syntax (for code), and Completeness.

3. **Determine Score (0.0 to 1.0)**:
   - **1.0 (Correct)**: Perfect / Fully Correct. Meets all requirements. (e.g., 5/5 or 10/10).
   - **0.5 (Partial)**: Partially Correct. Logic is okay but has minor errors/typos, OR covers ~50% of the requirements. (e.g., 2.5/5).
   - **0.0 (Incorrect)**: Wrong / Irrelevant / Syntax Errors / hallucinations. (e.g., 0/5).
   - *Note*: If specific marks were used, map the ratio to the nearest bucket (0.0, 0.25, 0.5, 0.75, 1.0).

### DETERMINISTIC RULES (NON-NEGOTIABLE):
- **Safety First**: If code has Syntax Errors or Security Risks -> AUTOMATIC 0.0.
- **Consistency**: The same input MUST yield the same score. Do not be "generous" or "random".
- **Relevance**: If answer is unrelated to the question -> AUTOMATIC 0.0.
- **No Hallucination**: Do not invent criteria not present in the description.

### FEEDBACK REQUIREMENTS:
- Start with "Score: X/Y" (if points were found) or "Assessment: [Status]".
- Provide the EXACT Correct Answer.
- Explain precisely WHY points were deducted (mention specific missing keywords, lines of code, or logic errors)."""
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=EvalDetail.model_json_schema(),
        )
        
        # Use fixed model
        original_model = self.model
        self.model = DeterministicEvalConfig.FIXED_MODEL
        
        try:
            # Consensus mechanism: 3 parallel calls with deterministic tiebreaker
            if DeterministicEvalConfig.USE_CONSENSUS and DeterministicEvalConfig.CONSENSUS_CALLS >= 2:
                async def _single_eval_call():
                    return await self._call_gemini_core(prompt, config, EvalDetail, "Question Evaluation")

                # Fire parallel calls
                logger.info(f"🔄 Running consensus evaluation ({DeterministicEvalConfig.CONSENSUS_CALLS} parallel calls)")
                results = await asyncio.gather(*[_single_eval_call() for _ in range(DeterministicEvalConfig.CONSENSUS_CALLS)])

                # Filter successful responses
                valid_responses = [r["response"] for r in results if r["success"] and "response" in r]

                if not valid_responses:
                    # All failed
                    return results[0]

                # Deterministic voting with tie-breaker
                votes = []
                for resp in valid_responses:
                    # Normalize to 0.0, 0.5, or 1.0
                    if resp.is_correct:
                        score = 1.0
                    elif resp.partial_credit is not None:
                        pc = float(resp.partial_credit)
                        # Quantize to nearest bucket
                        if pc >= 0.75:
                            score = 1.0
                        elif pc >= 0.25:
                            score = 0.5
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                    votes.append(score)

                from collections import Counter
                vote_counts = Counter(votes)
                
                # Majority voting with deterministic tie-breaker
                sorted_votes = vote_counts.most_common()
                
                if len(sorted_votes) > 1 and sorted_votes[0][1] == sorted_votes[1][1]:
                    # Tie scenario: select highest score (most lenient)
                    winner_score = max(sorted_votes[0][0], sorted_votes[1][0])
                    logger.warning(f"⚠️ Consensus TIE detected. Votes: {votes}. Tiebreaker: selecting {winner_score}")
                else:
                    winner_score = sorted_votes[0][0]

                # Find first response matching winner score
                for idx, resp in enumerate(valid_responses):
                    s = 1.0 if resp.is_correct else (float(resp.partial_credit) if resp.partial_credit is not None else 0.0)
                    # Quantize check
                    if s >= 0.75 and winner_score == 1.0:
                        resp_consensus = {"success": True, "response": resp.model_dump()}
                        EvaluationCache.set(content_hash, resp_consensus, eval_type="qa_evaluation")
                        logger.info(f"✓ Consensus Result: {votes} -> Winner: {winner_score} (Call #{idx+1})")
                        return resp_consensus
                    elif 0.25 <= s < 0.75 and winner_score == 0.5:
                        resp_consensus = {"success": True, "response": resp.model_dump()}
                        EvaluationCache.set(content_hash, resp_consensus, eval_type="qa_evaluation")
                        logger.info(f"✓ Consensus Result: {votes} -> Winner: {winner_score} (Call #{idx+1})")
                        return resp_consensus
                    elif s < 0.25 and winner_score == 0.0:
                        resp_consensus = {"success": True, "response": resp.model_dump()}
                        EvaluationCache.set(content_hash, resp_consensus, eval_type="qa_evaluation")
                        logger.info(f"✓ Consensus Result: {votes} -> Winner: {winner_score} (Call #{idx+1})")
                        return resp_consensus
                
                # Fallback: return first response
                resp_consensus = {"success": True, "response": valid_responses[0].model_dump()}
                EvaluationCache.set(content_hash, resp_consensus, eval_type="qa_evaluation")
                return resp_consensus
            else:
                # Single call (if consensus disabled)
                result = await self._call_gemini_core(prompt, config, EvalDetail, "Question Evaluation")
                if result["success"]:
                    result["response"] = result["response"].model_dump()
                    EvaluationCache.set(content_hash, result, eval_type="qa_evaluation")
                return result

        finally:
            self.model = original_model

    async def evaluate_ppt_structured(self, title: str, description: str, total_slides: int, slides_text: str) -> Dict:
        """Evaluate PPT Content - DETERMINISTIC"""
        # Content hash for caching
        content_hash = DeterministicEvalConfig.get_content_hash(f"{title}|||{description}|||{slides_text}")
        
        cached_result = EvaluationCache.get(content_hash, eval_type="ppt_content")
        if cached_result is not None:
            return cached_result
        
        # Standardized prompt for PPT evaluation
        prompt = f"""### ROLE: You are a professional presentation evaluator.
Evaluate the PowerPoint presentation content based on the assignment requirements.

### ASSIGNMENT TITLE:
{title}

### ASSIGNMENT REQUIREMENTS/DESCRIPTION:
{description}

### PRESENTATION METADATA:
Total Slides: {total_slides}

### SLIDE CONTENT:
{slides_text}

### EVALUATION CRITERIA (DETERMINISTIC RUBRIC):
1. **Content Quality (0-100)**: How well the slides address the requirements, accuracy, completeness
2. **Structure (0-100)**: Logical flow, organization, coherence between slides
3. **Alignment (0-100)**: How well content aligns with assignment description and requirements

### SCORING RULES:
- 90-100: Excellent - Fully meets all criteria with high quality
- 75-89: Good - Meets most criteria with minor gaps
- 60-74: Satisfactory - Meets basic criteria with some issues
- 45-59: Needs Improvement - Missing key elements or significant issues
- 0-44: Inadequate - Does not meet most criteria

### OUTPUT REQUIREMENTS:
- Provide exact scores (0-100) for each criterion
- List specific strengths (2-4 items)
- List improvement areas (2-4 items)
- Provide concise overall summary (2-3 sentences)"""
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=PPTEvaluation.model_json_schema(),
        )
        
        result = await self._call_gemini_core(prompt, config, PPTEvaluation, "PPT Evaluation")
        if result["success"]:
            result["response"] = result["response"].model_dump()
            EvaluationCache.set(content_hash, result, eval_type="ppt_content")
        return result

    async def evaluate_ppt_design_structured(self, design_description: str, filename: str, total_slides: int) -> Dict:
        """Evaluate PPT Design - DETERMINISTIC"""
        content_hash = DeterministicEvalConfig.get_content_hash(f"{design_description}|||{filename}|||{total_slides}")
        
        cached_result = EvaluationCache.get(content_hash, eval_type="ppt_design")
        if cached_result is not None:
            return cached_result
        
        # Standardized design evaluation prompt
        prompt = f"""### ROLE: You are a professional design evaluator.
Evaluate the PowerPoint design based on visual quality and professional standards.

### FILE: {filename}
### TOTAL SLIDES: {total_slides}

### DESIGN METADATA/DESCRIPTION:
{design_description}

### EVALUATION CRITERIA (DETERMINISTIC DESIGN RUBRIC):
1. **Visual Clarity (0-100)**: Text readability, image quality, visual elements are clear
2. **Layout Balance (0-100)**: Proper spacing, alignment, use of white space
3. **Color Consistency (0-100)**: Consistent color scheme, proper contrast, professional palette
4. **Typography (0-100)**: Font choices, size consistency, hierarchy clarity
5. **Visual Appeal (0-100)**: Overall aesthetics, professional appearance, visual engagement

### SCORING RULES:
- 90-100: Excellent design - Professional, polished, highly visually appealing
- 75-89: Good design - Clean layout, mostly consistent, professional
- 60-74: Acceptable design - Basic standards met with minor issues
- 45-59: Needs work - Inconsistent or unprofessional elements present
- 0-44: Poor design - Significant visual or professional issues

### OUTPUT REQUIREMENTS:
- Provide exact scores (0-100) for each design criterion
- List specific design strengths (2-3 items)
- List design improvements (2-3 items)
- Provide concise design summary (2-3 sentences)"""
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=PPTDesignEvaluation.model_json_schema(),
        )
        
        result = await self._call_gemini_core(prompt, config, PPTDesignEvaluation, "PPT Design Evaluation")
        if result["success"]:
            result["response"] = result["response"].model_dump()
            EvaluationCache.set(content_hash, result, eval_type="ppt_design")
        return result

    async def evaluate_ppt_design_vision_structured(self, slide_images_base64: List[str]) -> Dict:
        """Evaluate PPT design via vision - DETERMINISTIC"""
        # Hash the image list for caching
        import hashlib
        image_hash = hashlib.sha256(str(len(slide_images_base64)).encode()).hexdigest()
        
        cached_result = EvaluationCache.get(image_hash, eval_type="ppt_vision")
        if cached_result is not None:
            return cached_result
        
        parts = ["### ROLE: You are a professional design evaluator.\nEvaluate the design and visual quality of these PowerPoint slides based on professional presentation standards.\n\nEVALUATION CRITERIA:\n1. Visual Clarity\n2. Layout Balance\n3. Color Consistency\n4. Typography\n5. Overall Visual Appeal\n\nProvide scores 0-100 for each criterion and professional feedback."]
        
        for img_base64 in slide_images_base64:
            try:
                import base64 as b64_module
                img_bytes = b64_module.b64decode(img_base64)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            except Exception as e:
                logger.error(f"Error decoding slide image: {e}")
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=PPTDesignEvaluation.model_json_schema(),
        )
        
        result = await self._call_gemini_core(parts, config, PPTDesignEvaluation, "PPT Vision Design Evaluation")
        if result["success"]:
            result["response"] = result["response"].model_dump()
            EvaluationCache.set(image_hash, result, eval_type="ppt_vision")
        return result

    async def evaluate_git_repository_structured(self, prompt: str) -> Dict:
        """Evaluate Git Repository - DETERMINISTIC"""
        content_hash = DeterministicEvalConfig.get_content_hash(prompt)
        
        cached_result = EvaluationCache.get(content_hash, eval_type="git_analysis")
        if cached_result is not None:
            return cached_result
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=GitProjectInfo.model_json_schema(),
        )
        
        result = await self._call_gemini_core(prompt, config, GitProjectInfo, "Git Repo Analysis")
        if result["success"]:
            result["response"] = result["response"].model_dump()
            EvaluationCache.set(content_hash, result, eval_type="git_analysis")
        return result

    async def grade_git_repository_structured(self, prompt: str) -> Dict:
        """Grade Git Repository - DETERMINISTIC"""
        content_hash = DeterministicEvalConfig.get_content_hash(prompt)
        
        cached_result = EvaluationCache.get(content_hash, eval_type="git_grading")
        if cached_result is not None:
            return cached_result
        
        config = types.GenerateContentConfig(
            temperature=DeterministicEvalConfig.TEMPERATURE,
            response_mime_type="application/json",
            response_schema=GitGradingResult.model_json_schema(),
        )
        
        result = await self._call_gemini_core(prompt, config, GitGradingResult, "Git Repo Grading")
        if result["success"]:
            result["response"] = result["response"].model_dump()
            EvaluationCache.set(content_hash, result, eval_type="git_grading")
        return result

    async def generate(self, prompt: str, model: Optional[str] = None, system_message: Optional[str] = None, temperature: float = 0.0) -> Dict:
        config = types.GenerateContentConfig(
            system_instruction=system_message or "You are a professional assistant.",
            temperature=temperature,
            max_output_tokens=50000,
        )
        return await self._call_gemini_core(prompt, config, None, "Generate Text")

    async def generate_with_images(self, messages: List[Dict], model: Optional[str] = None, system_message: Optional[str] = None, temperature: float = 0.0) -> Dict:
        # Compatibility wrapper for image messages
        parts = []
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text': parts.append(item.get('text'))
                    elif item.get('type') == 'image_url':
                        img_url = item.get('image_url', {}).get('url', '')
                        if img_url.startswith('data:image'):
                            try:
                                import base64 as b64_module
                                header, data_b64 = img_url.split(',', 1)
                                mime_type = header.split(';')[0].split(':')[1]
                                img_bytes = b64_module.b64decode(data_b64)
                                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                            except Exception: pass
            else: parts.append(content)
        
        config = types.GenerateContentConfig(
            system_instruction=system_message or "Analyzes slide images.",
            temperature=temperature,
            max_output_tokens=50000,
        )
        return await self._call_gemini_core(parts, config, None, "Vision Generation")

    def check_connection(self) -> bool:
        """Compatibility check for LLM service status"""
        client = self._get_client()
        return client is not None

    def list_models(self) -> List[str]:
        """Compatibility list models (returns current configured model)"""
        return [self.model]
