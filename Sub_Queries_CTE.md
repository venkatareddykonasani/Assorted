# Subqueries

Subqueries, also known as inner queries or nested queries, are SQL queries nested inside another SQL query. They are used to perform operations that depend on the result set of another query. Subqueries can be used in various parts of an SQL statement, including the SELECT, FROM, WHERE, and HAVING clauses.

### Types of Subqueries
1. **Single-row subqueries**: Return only one row.
2. **Multiple-row subqueries**: Return more than one row.
3. **Multiple-column subqueries**: Return more than one column.

### Example 1: Single-row Subquery
This example demonstrates a subquery in the `WHERE` clause to find employees who earn more than the average salary in the company.

```sql
SELECT employee_id, first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```
In this example:
- The inner query `(SELECT AVG(salary) FROM employees)` calculates the average salary.
- The outer query retrieves employees whose salaries are greater than the average salary.

### Example 2: Multiple-row Subquery
This example uses a subquery in the `IN` clause to find employees who work in departments located in New York.

```sql
SELECT employee_id, first_name, last_name, department_id
FROM employees
WHERE department_id IN (SELECT department_id
                        FROM departments
                        WHERE location = 'New York');
```
In this example:
- The inner query `(SELECT department_id FROM departments WHERE location = 'New York')` retrieves the department IDs of departments located in New York.
- The outer query retrieves employees who work in those departments.

### Example 3: Correlated Subquery
A correlated subquery references columns from the outer query and is re-evaluated for each row processed by the outer query. This example finds employees who earn more than the average salary in their respective departments.

```sql
SELECT employee_id, first_name, last_name, salary, department_id
FROM employees e1
WHERE salary > (SELECT AVG(salary)
                FROM employees e2
                WHERE e1.department_id = e2.department_id);
```
In this example:
- The inner query `(SELECT AVG(salary) FROM employees e2 WHERE e1.department_id = e2.department_id)` calculates the average salary for each department.
- The outer query retrieves employees who earn more than the average salary in their respective departments.

### Example 4: Subquery in the FROM Clause
This example demonstrates using a subquery in the `FROM` clause to find the department with the highest average salary.

```sql
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) = (SELECT MAX(avg_salary)
                      FROM (SELECT department_id, AVG(salary) AS avg_salary
                            FROM employees
                            GROUP BY department_id) dept_avg);
```
In this example:
- The inner subquery `(SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id)` calculates the average salary for each department.
- The outer subquery `(SELECT MAX(avg_salary) FROM (...))` finds the maximum average salary.
- The main query retrieves the department with that maximum average salary.

These examples demonstrate the versatility and power of subqueries in SQL for performing complex queries and data analysis.

---

# CTEs
Common Table Expressions (CTEs) in SQL, also known as the `WITH` clause, are a way to define temporary result sets that can be referenced within a `SELECT`, `INSERT`, `UPDATE`, or `DELETE` statement. CTEs improve the readability and maintainability of complex queries by breaking them into more manageable parts.

### Syntax of CTE
```sql
WITH cte_name (column1, column2, ...)
AS (
    -- CTE query
    SELECT ...
)
-- Main query using the CTE
SELECT column1, column2, ...
FROM cte_name;
```

### Example 1: Simple CTE
This example uses a CTE to calculate the average salary of employees and then find employees who earn more than the average salary.

```sql
WITH AverageSalary AS (
    SELECT AVG(salary) AS avg_salary
    FROM employees
)
SELECT employee_id, first_name, last_name, salary
FROM employees, AverageSalary
WHERE employees.salary > AverageSalary.avg_salary;
```
In this example:
- The CTE `AverageSalary` calculates the average salary of all employees.
- The main query retrieves employees whose salaries are greater than the average salary.

### Example 2: Recursive CTE
Recursive CTEs are useful for hierarchical or tree-structured data, such as organizational charts or bill of materials. This example finds the hierarchical structure of an organization's employees.

```sql
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT employee_id, first_name, last_name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.first_name, e.last_name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
)
SELECT employee_id, first_name, last_name, manager_id, level
FROM EmployeeHierarchy
ORDER BY level, manager_id;
```
In this example:
- The anchor member of the CTE selects the top-level employees (those with no manager).
- The recursive member joins the `employees` table with the `EmployeeHierarchy` CTE to find employees reporting to each manager, incrementing the hierarchy level.
- The main query retrieves and orders the hierarchical structure of the employees.

### Example 3: CTE with Multiple Subqueries
This example demonstrates using multiple CTEs to find departments with an average salary above a certain threshold.

```sql
WITH DepartmentSalaries AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
HighSalaryDepartments AS (
    SELECT department_id
    FROM DepartmentSalaries
    WHERE avg_salary > 70000
)
SELECT d.department_id, d.department_name
FROM departments d
INNER JOIN HighSalaryDepartments hsd ON d.department_id = hsd.department_id;
```
In this example:
- The first CTE `DepartmentSalaries` calculates the average salary for each department.
- The second CTE `HighSalaryDepartments` filters out departments with an average salary above 70,000.
- The main query retrieves the names of those departments by joining with the `departments` table.

CTEs provide a powerful way to organize complex queries, making them easier to read, write, and maintain. They are especially useful when the same subquery needs to be referenced multiple times in a query or when working with hierarchical data.
