# Contribution Guidelines

Thank you for helping keep this list useful. The list is deliberately curated: it is not a directory of everything, but a short list of projects you can safely bet a production system on.

## Inclusion criteria

A project must meet **all** of these to be added:

1. **Open source.** The project carries an [OSI-approved license](https://opensource.org/licenses), and the core is genuinely usable without a commercial tier. Closed-source platforms with open-source client SDKs do not qualify.
2. **Actively maintained.** A release or meaningful commit within the last 3 months. Archived, frozen, or maintenance-mode projects are removed when discovered (a monthly freshness audit checks this automatically).
3. **Proven in production.** Evidence of real-world adoption at scale: named production users, significant sustained download numbers, or inclusion in major cloud/vendor offerings. Research prototypes, pre-1.0 experiments, and personal projects do not qualify, however promising.
4. **Not redundant.** The project adds a capability the list doesn't already cover, or is clearly stronger than an existing entry in the same slot.

## Adding a project

- Add one table row in the appropriate section, keeping **alphabetical order** (case-insensitive, by project name).
- Use the 4-column format: `| [Name](https://github.com/org/repo) | Support-Org | Tag1, Tag2 | Description |`.
- Keep the description under 120 characters, factual, and free of marketing language.
- Link to the project's **current** repository (no redirects from renamed or moved repos).
- Run the checks locally before opening the PR:

  ```sh
  python3 scripts/check_readme.py
  python3 scripts/format_tables.py
  npx awesome-lint
  ```

## Disclosure

If you are affiliated with the project you are adding (author, maintainer, employee), say so in the PR description. Affiliated submissions are welcome but are held to the same adoption evidence as everything else; PRs without evidence of third-party production use will be closed.

## Removals and recategorization

Projects that become archived, unmaintained, or closed-source are removed. If you think an entry is in the wrong category, or should be removed, open an issue or PR with your reasoning.
