
import tiktoken

x = """ ('Title: Software Engineer II Compliance at Chainalysis\nLocation: Remote\n

Job Objective: The Compliance organization is focused on growing the crypto ecosystem by simplifying the work needed for compliance and risk management at a massive scale. Their goal is to ensure developers can send data for any crypto asset and network scale their systems to handle increasing volumes of data and provide meaningful insights to customers.\n

Responsibilities/Key duties: Backend engineers will be critical in building and scaling the APIs and data layers that customers rely on to stop crime, understand risk, and strategize about their business. They will work alongside infrastructure and security-focused engineers to make services highly available and safe for customers to use for their most sensitive and real-time blockchain workflows.

\nQualifications/Requirements: Designed and implemented microservices-based systems in a major cloud provider like AWS or GCP. Experience with object-oriented programming languages, particularly Java. 

\nPreferred Skills/Experience: A bias to ship and iterate alongside product management and design partners. Exposure to or interest in the cryptocurrency technology ecosystem.

\nAbout the company: Chainalysis is a company focused on simplifying compliance and risk management in the crypto ecosystem. They provide services and insights to help customers navigate the cryptocurrency landscape.

\nCompensation and Benefits: Not provided in the given information.'"""


def num_tokens(text: str, model: str ="gpt-3.5-turbo") -> int:
	#Return the number of tokens in a string.
	encoding = tiktoken.encoding_for_model(model)
	return len(encoding.encode(text))

count = num_tokens(x)

print(count)