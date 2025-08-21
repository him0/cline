/**
 * Formats rule data for export to markdown
 */

export interface ExportRulesData {
	globalRules?: string
	localRules?: string
	cursorRules?: string
	cursorRulesDir?: string
	windsurfRules?: string
	clineIgnore?: string
	preferredLanguage?: string
}

/**
 * Formats rules data into markdown sections for export
 * @param rulesData The rules data to format
 * @returns Formatted markdown string or empty string if no rules
 */
export function formatRulesForExport(rulesData: ExportRulesData): string {
	const sections: string[] = []

	// Add preferred language if specified
	if (rulesData.preferredLanguage) {
		sections.push("## Preferred Language\n\n" + rulesData.preferredLanguage)
	}

	// Add global Cline rules
	if (rulesData.globalRules) {
		sections.push("## Global Cline Rules\n\n" + rulesData.globalRules)
	}

	// Add local Cline rules
	if (rulesData.localRules) {
		sections.push("## Local Cline Rules\n\n" + rulesData.localRules)
	}

	// Add Cursor rules file
	if (rulesData.cursorRules) {
		sections.push("## Cursor Rules File\n\n" + rulesData.cursorRules)
	}

	// Add Cursor rules directory
	if (rulesData.cursorRulesDir) {
		sections.push("## Cursor Rules Directory\n\n" + rulesData.cursorRulesDir)
	}

	// Add Windsurf rules
	if (rulesData.windsurfRules) {
		sections.push("## Windsurf Rules\n\n" + rulesData.windsurfRules)
	}

	// Add Cline ignore patterns
	if (rulesData.clineIgnore) {
		sections.push("## Cline Ignore Patterns\n\n```\n" + rulesData.clineIgnore + "\n```")
	}

	// Return formatted sections or empty string
	if (sections.length === 0) {
		return ""
	}

	return "# Configuration and Rules\n\n" + sections.join("\n\n") + "\n\n"
}
