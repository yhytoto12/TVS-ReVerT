from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator, Tuple, Union, AsyncIterator
import os
import sys
import re
import argparse
import asyncio
import openai
import torch
from transformers import AutoTokenizer

# Set VLLM environment
os.environ["VLLM_USE_V1"] = "0"
from vllm import LLM, SamplingParams

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts import (
    SYSTEM_PROMPT_FOR_THINK,
    SYSTEM_PROMPT_FOR_VERBALIZER,
    BOV_TOKEN,
    EOV_TOKEN,
    CON_TOKEN
)

# Custom print for verbalize output (ANSI escape codes)
def cprint(text, **kwargs):
    print(f"\033[92m{text}\033[0m", **kwargs)


class AllowedTokensLogitsProcessor:
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # scores: [batch, vocab_size]
        mask = torch.full_like(scores, float('-inf'))
        mask[list(self.allowed_token_ids)] = 0
        scores = scores + mask
        return scores


class ThinkModelClient:
    """Client class for handling async streaming output from a served think model"""

    def __init__(self, model_name_or_path: str, use_openai: bool = False, vllm_url: str = "http://localhost:8000/v1"):
        self.model_name_or_path = model_name_or_path
        self.use_openai = use_openai

        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai.AsyncOpenAI(api_key=api_key)
        else:
            # Connect to a local VLLM server (OpenAI-compatible endpoint)
            self.client = openai.AsyncOpenAI(
                base_url=vllm_url,
                api_key="EMPTY",
            )

    async def generate_streaming(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Generate think model response via async streaming"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                stream=True,
                temperature=0.1,
                max_tokens=2048,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"API error: {e}")
            yield ""


class VerbalizerClient:
    """Client class for handling streaming output from verbalizer model"""

    def __init__(self, model_name_or_path: str):
        self.tokenizer, self.model = load_model_vllm(model_name_or_path, utilization=0.5)

        # Special tokens
        self.BOV_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(BOV_TOKEN)
        self.EOV_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(EOV_TOKEN)
        self.CON_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(CON_TOKEN)

        # Sampling parameters
        self.sampling_params_for_bov = SamplingParams(
            max_tokens=1,
            skip_special_tokens=False,
            temperature=0.0,
            logits_processors=[AllowedTokensLogitsProcessor([self.BOV_TOKEN_ID, self.CON_TOKEN_ID])],
        )

        self.sampling_params_for_eov = SamplingParams(
            max_tokens=512,
            skip_special_tokens=False,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.EOV_TOKEN_ID,
            ],
        )

    def init_context(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_VERBALIZER},
            {"role": "user", "content": question},
        ]

        init_context = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return init_context

    def decide_mode(self, context: str, current_segment: str) -> Tuple[bool, str]:
        """
        Determine whether to start verbalizing based on the current segment.
        """
        input_text = context + current_segment
        outputs = self.model.generate(input_text, self.sampling_params_for_bov, use_tqdm=False)
        next_token = outputs[0].outputs[0].text

        if next_token == BOV_TOKEN:
            context = input_text + BOV_TOKEN
            return True, context
        elif next_token == CON_TOKEN:
            context = input_text
            return False, context
        else:
            raise ValueError(f"Unexpected token generated: {next_token}")

    def verbalize_streaming(self, context: str) -> Tuple[str, str]:
        """
        Generate verbalize results.
        Returns: verbalized text and updated context
        """
        outputs = self.model.generate(context, self.sampling_params_for_eov, use_tqdm=False)
        verbalized_text = outputs[0].outputs[0].text
        context = context + verbalized_text + EOV_TOKEN
        return verbalized_text, context


def load_model_vllm(model_name_or_path, utilization=0.5) -> Tuple[AutoTokenizer, LLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = LLM(
        model=model_name_or_path,
        dtype="bfloat16",
        gpu_memory_utilization=utilization,
        tensor_parallel_size=1,
        max_model_len=4096,
    )
    return tokenizer, model


async def interactive_demo(think_client: ThinkModelClient, verbalizer_client: VerbalizerClient):
    """Interactive demo function"""
    print("\n=== ReVerT Interactive Demo (Async) ===")
    print("Enter your question (or 'quit' to exit):")
    loop = asyncio.get_event_loop()

    while True:
        question = await loop.run_in_executor(None, lambda: input("\nQuestion: ").strip())
        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        # Think stage
        print("\n🤔 Starting Think stage...")
        think_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_THINK},
            {"role": "user", "content": question}
        ]

        context = await loop.run_in_executor(None, verbalizer_client.init_context, question)

        think_content = ""
        verbalize_content = ""
        buffer = ""
        async for chunk in think_client.generate_streaming(think_messages):
            think_content += chunk
            print(chunk, end="", flush=True)
            buffer += chunk

            while "\n" in buffer:
                segments = buffer.split("\n", 1)
                current_segment = segments[0] + "\n"
                buffer = segments[1]

                # Run blocking verbalizer calls in a thread
                start_verbalizing, context = await loop.run_in_executor(
                    None, verbalizer_client.decide_mode, context, current_segment
                )

                if start_verbalizing:
                    verbalized_text, context = await loop.run_in_executor(
                        None, verbalizer_client.verbalize_streaming, context
                    )
                    cprint(f"\n💬 verbalize: {verbalized_text}\n")
                    verbalize_content += verbalized_text + " "

        # Process any remaining buffer content
        if buffer:
            # If there's remaining buffer, always do verbalization
            context = context + buffer.strip() + "\n" + BOV_TOKEN
            verbalized_text, context = await loop.run_in_executor(
                None, verbalizer_client.verbalize_streaming, context
            )
            cprint(f"\n\n💬 verbalize: {verbalized_text}\n")
            verbalize_content += verbalized_text + " "

        print(f"=== Think content ===")
        print(f"{think_content}\n")
        cprint(f"=== Verbalize content ===")
        cprint(f"{verbalize_content}\n")


async def main():
    parser = argparse.ArgumentParser(description="ReVerT Interactive Demo (Async)")
    parser.add_argument("--think_model", type=str, required=True, help="Name of the served think model")
    parser.add_argument("--verbalizer_model", type=str, required=True, help="Path to verbalizer model")
    parser.add_argument("--use_openai_think", action="store_true", help="Use OpenAI API for think model")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="URL of the VLLM server")

    args = parser.parse_args()

    # Initialize models
    print("Loading models...")
    think_client = ThinkModelClient(
        args.think_model,
        use_openai=args.use_openai_think,
        vllm_url=args.vllm_url,
    )
    # Verbalizer client is sync and loads model locally
    verbalizer_client = VerbalizerClient(args.verbalizer_model)
    print("Model loading completed!")

    await interactive_demo(think_client, verbalizer_client)

if __name__ == "__main__":
    asyncio.run(main())
