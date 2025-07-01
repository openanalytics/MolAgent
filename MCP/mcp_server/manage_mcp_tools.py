from typing import Any, Dict, List, Optional, TypeVar

from smolagents import (
    Tool,
    CodeAgent,
    ToolCallingAgent,
    ToolCollection
)
from mcpadapt.core import MCPAdapt
from mcpadapt.smolagents_adapter import SmolAgentsAdapter

class MCPServerControl():
    def __init__(self, server_urls:List[str]=None):
        self.adapters=[]
        self.tools=[]
        for url in server_urls:
            adapter = MCPAdapt(
                #{"url": url, "timeout": 30, "sse_read_timeout": 30},
                {"url": url, "timeout": 1e60, "sse_read_timeout": 1e60},
                #{"url": url},
                SmolAgentsAdapter(),
                connect_timeout=10,
            )
            self.adapters.append(adapter)
            adapter.__enter__()
            serv_tools: list[Tool] = adapter.tools()
            for t in serv_tools:
                self.tools.append(t)
    def get_tools(self):
        return self.tools
    
    def close(self):
        for adapt in self.adapters:
            # close when you do not need anymore
            adapt.__exit__(None, None, None)            
            
#tdc_tool = ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/sse"}, trust_remote_code=True)

#from mcpadapt.core import MCPAdapt
#from mcpadapt.smolagents_adapter import SmolAgentsAdapter
