# Run agent and force single-step behavior
def run_agent(input_text: str):
    try:
        result = agent.run(input_text)
        # Extract intermediate steps if they exist
        if isinstance(result, dict):
            if "intermediate_steps" in result:
                last_step = result["intermediate_steps"][-1]
                if isinstance(last_step, tuple) and len(last_step) >= 2:
                    return last_step[1]  # Return final output
            return result.get("output", "No direct output found.")
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error in processing."

# Test the tool
if __name__ == "__main__":
    input_text = "I am excited to run this code and convert to uppercase."
    final_output = run_agent(input_text)
    print(f"Final Output: {final_output}")
