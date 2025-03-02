import { Anthropic } from "@anthropic-ai/sdk"
import axios from "axios"

import { ApiHandlerOptions, ModelInfo, liteLlmDefaultModelId, liteLlmModelInfoSaneDefaults } from "../../shared/api"
import { ApiHandler, SingleCompletionHandler } from "../index"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream, ApiStreamUsageChunk } from "../transform/stream"

export interface LiteLlmHandlerOptions extends ApiHandlerOptions {
	defaultHeaders?: Record<string, string>
}

export class LiteLlmHandler implements ApiHandler, SingleCompletionHandler {
	protected options: LiteLlmHandlerOptions

	constructor(options: LiteLlmHandlerOptions) {
		this.options = options
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelInfo = this.getModel().info
		const baseUrl = this.options.liteLlmBaseUrl ?? "http://localhost:4000"
		const apiKey = this.options.liteLlmApiKey ?? "noop"
		const modelId = this.options.liteLlmModelId ?? liteLlmDefaultModelId

		const systemMessage = {
			role: "system",
			content: systemPrompt,
		}

		const convertedMessages = [systemMessage, ...convertToOpenAiMessages(messages)]

		try {
			const response = await axios.post(
				`${baseUrl}/chat/completions`,
				{
					model: modelId,
					messages: convertedMessages,
					temperature: this.options.modelTemperature ?? 0,
					max_tokens: this.options.modelMaxTokens ?? modelInfo.maxTokens,
					stream: false,
				},
				{
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${apiKey}`,
						...this.options.defaultHeaders,
					},
				},
			)

			const content = response.data.choices[0]?.message?.content || ""
			yield {
				type: "text",
				text: content,
			}

			if (response.data.usage) {
				yield this.processUsageMetrics(response.data.usage)
			}
		} catch (error) {
			console.error("LiteLLM API error:", error)
			if (axios.isAxiosError(error)) {
				yield {
					type: "text",
					text: `LiteLLM API error: ${error.response?.data?.error?.message || error.message}`,
				}
			} else {
				yield {
					type: "text",
					text: `LiteLLM API error: ${(error as Error).message}`,
				}
			}
		}
	}

	protected processUsageMetrics(usage: any): ApiStreamUsageChunk {
		return {
			type: "usage",
			inputTokens: usage?.prompt_tokens || 0,
			outputTokens: usage?.completion_tokens || 0,
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.liteLlmModelId ?? liteLlmDefaultModelId,
			info: this.options.liteLlmModelInfo ?? liteLlmModelInfoSaneDefaults,
		}
	}

	async completePrompt(prompt: string): Promise<string> {
		try {
			const baseUrl = this.options.liteLlmBaseUrl ?? "http://localhost:4000"
			const apiKey = this.options.liteLlmApiKey ?? "noop"
			const modelId = this.options.liteLlmModelId ?? liteLlmDefaultModelId

			const response = await axios.post(
				`${baseUrl}/chat/completions`,
				{
					model: modelId,
					messages: [{ role: "user", content: prompt }],
					temperature: this.options.modelTemperature ?? 0,
				},
				{
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${apiKey}`,
						...this.options.defaultHeaders,
					},
				},
			)

			return response.data.choices[0]?.message?.content || ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`LiteLLM completion error: ${error.message}`)
			}
			throw error
		}
	}
}

export async function getLiteLlmModels(baseUrl?: string, apiKey?: string) {
	try {
		if (!baseUrl) {
			return []
		}

		if (!URL.canParse(baseUrl)) {
			return []
		}

		const config: Record<string, any> = {}

		if (apiKey) {
			config["headers"] = { Authorization: `Bearer ${apiKey}` }
		}

		const response = await axios.get(`${baseUrl}/models`, config)
		const modelsArray = response.data?.data?.map((model: any) => model.id) || []
		return [...new Set<string>(modelsArray)]
	} catch (error) {
		return []
	}
}
