# Third-party references

This project is an independent implementation. It does not vendor or import runtime code from the
repositories below. Their public architecture and prompts were inspected and compatible patterns
were reimplemented behind this project's existing safety contracts.

- XiYan-SQL, XGenerationLab, Apache-2.0, inspected commit
  `603dedac706d57ece47bc30d02f90744a537b6a0`. Adapted concepts: compact M-Schema-style context,
  multi-generator candidates, ICL candidate diversity, refinement, and selection.
- ReFoRCE, Snowflake-Labs, Apache-2.0, inspected commit
  `d2658991882f06fa0658bb7782ee9ea515fbb10b`. Adapted concepts: table-level schema linking,
  bounded execution feedback, result consensus, and conditional inspection.
- Arctic-Text2SQL-R1-7B, Snowflake, Apache-2.0 model. Supported as an optional local generator
  interface; model weights are not bundled or downloaded by this repository.

The untouched reference clones live outside this project in `/Users/meisam/Documents/text-to-sql/references`.

- Uber QueryGPT engineering article, published September 19, 2024. Adapted public concepts:
  workspaces, intent classification, editable table proposals, column pruning, prompt enhancement,
  and component-level evaluation. No Uber source code or internal data was provided or copied.
  <https://www.uber.com/de/en/blog/query-gpt/>
