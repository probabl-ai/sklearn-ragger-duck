BUTTON_PROCESSING = "Thinking"
BUTTON_TYPING = "Typing"
PROMPT = "(Enter your prompt here, type !help for a list of commands)"
BUTTON_SEND = "Ask the LLaMa"
BUTTON_WAIT = "Wait"
HELP = """<table style='margin-bottom:10px'>
<tr><th colspan=2>Server commands</th></tr>
<tr><td>!models</td><td>List available models</td></tr>
<tr><td>!model</td><td>Show currently loaded model</td></tr>
<tr><td>!model (filename)</td><td>Load a different model</td></tr>
<tr><td>!stop</td><td>List of currenlty set stop words</td></tr>
<tr><td>!stop ['word1',...]&nbsp;</td><td>Assign new stopwords</td></tr>
<tr><td>!system</td><td>System State (used/free CPU and RAM)</td></tr>
</table>
<table>
<tr><th colspan=2>Prompt templates (pres TAB to complete)</th></tr>
<tr><td>#vic</td><td>Helpful AI Vicuna 1.1 prompt template</td></tr>
<tr><td>#story&nbsp;</td><td>Storyteller Vicuna 1.1 prompt template</td></tr>
<tr><td>###</td><td>Instruct/Response prompt template</td></tr>
</table>"""
