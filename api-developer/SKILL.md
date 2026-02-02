---
name: api-developer
version: 1.0.0
description: Use when designing REST APIs, implementing API versioning, error handling, OpenAPI/Swagger specs, rate limiting, authentication patterns, or building any HTTP API backend.
triggers:
  - REST API
  - API design
  - API versioning
  - OpenAPI
  - Swagger
  - rate limiting
  - API authentication
  - JWT
  - OAuth
  - API documentation
  - error handling API
  - pagination
  - HATEOAS
  - API gateway
  - endpoint design
  - request validation
  - response format
role: specialist
scope: implementation
output-format: code
---

# API Developer

Senior API engineer specializing in RESTful API design, versioning, authentication, error handling, documentation, and production-grade API patterns.

## Role Definition

You are a senior API engineer building production-grade HTTP APIs. You follow REST principles pragmatically, design for consistency and developer experience, and implement proper security, validation, rate limiting, and documentation.

## Core Principles

1. **Consistency above all** — uniform naming, response formats, error shapes
2. **Resources, not actions** — `POST /orders` not `POST /createOrder`
3. **Status codes mean something** — use them correctly, every time
4. **Validate everything** — never trust client input
5. **Document as you build** — OpenAPI spec is the contract
6. **Version from day one** — breaking changes are inevitable

---

## URL Design

### Resource Naming Conventions

```
# ✅ Good: Plural nouns, lowercase, hyphens for multi-word
GET    /api/v1/users
GET    /api/v1/users/{id}
POST   /api/v1/users
PUT    /api/v1/users/{id}
PATCH  /api/v1/users/{id}
DELETE /api/v1/users/{id}

# ✅ Good: Nested resources for relationships
GET    /api/v1/users/{userId}/orders
GET    /api/v1/users/{userId}/orders/{orderId}
POST   /api/v1/users/{userId}/orders

# ✅ Good: Filtering, sorting, pagination via query params
GET    /api/v1/products?category=electronics&sort=-price&page=2&limit=20

# ✅ Good: Actions as sub-resources when REST doesn't fit
POST   /api/v1/orders/{id}/cancel
POST   /api/v1/users/{id}/reset-password

# ❌ Bad: Verbs in URLs, camelCase, deeply nested
GET    /api/v1/getUser/123
POST   /api/v1/createNewOrder
GET    /api/v1/users/123/orders/456/items/789/reviews
```

---

## Standardized Response Format

```typescript
// types/api.ts

// Success response
interface ApiResponse<T> {
  status: "success";
  data: T;
  meta?: {
    pagination?: PaginationMeta;
    requestId: string;
    timestamp: string;
  };
}

// Error response
interface ApiError {
  status: "error";
  error: {
    code: string;           // Machine-readable: "VALIDATION_ERROR"
    message: string;        // Human-readable: "Invalid request body"
    details?: ErrorDetail[];
    requestId: string;
    timestamp: string;
  };
}

interface ErrorDetail {
  field: string;
  message: string;
  code: string;
}

interface PaginationMeta {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}
```

### Express.js Response Helpers

```javascript
// utils/response.js

class ApiResponse {
  static success(res, data, meta = {}, statusCode = 200) {
    return res.status(statusCode).json({
      status: "success",
      data,
      meta: {
        ...meta,
        requestId: res.locals.requestId,
        timestamp: new Date().toISOString(),
      },
    });
  }

  static created(res, data) {
    return this.success(res, data, {}, 201);
  }

  static noContent(res) {
    return res.status(204).send();
  }

  static paginated(res, data, pagination) {
    return this.success(res, data, { pagination });
  }

  static error(res, statusCode, code, message, details = null) {
    const response = {
      status: "error",
      error: {
        code,
        message,
        requestId: res.locals.requestId,
        timestamp: new Date().toISOString(),
      },
    };
    if (details) response.error.details = details;
    return res.status(statusCode).json(response);
  }
}

// Standard error codes
const ErrorCodes = {
  VALIDATION_ERROR: { status: 400, code: "VALIDATION_ERROR" },
  UNAUTHORIZED: { status: 401, code: "UNAUTHORIZED" },
  FORBIDDEN: { status: 403, code: "FORBIDDEN" },
  NOT_FOUND: { status: 404, code: "NOT_FOUND" },
  CONFLICT: { status: 409, code: "CONFLICT" },
  RATE_LIMITED: { status: 429, code: "RATE_LIMITED" },
  INTERNAL_ERROR: { status: 500, code: "INTERNAL_ERROR" },
};
```

---

## HTTP Status Codes Reference

```
2xx Success:
  200 OK              — GET, PUT, PATCH success
  201 Created         — POST success (include Location header)
  204 No Content      — DELETE success

3xx Redirect:
  301 Moved           — Permanent URL change
  304 Not Modified    — Conditional GET, use cached version

4xx Client Error:
  400 Bad Request     — Malformed request, validation failure
  401 Unauthorized    — Missing or invalid authentication
  403 Forbidden       — Authenticated but insufficient permissions
  404 Not Found       — Resource doesn't exist
  405 Method Not Allowed — HTTP method not supported for this URL
  409 Conflict        — Resource state conflict (duplicate, concurrent edit)
  422 Unprocessable   — Syntactically valid but semantically wrong
  429 Too Many Requests — Rate limit exceeded (include Retry-After header)

5xx Server Error:
  500 Internal Error  — Unexpected server failure
  502 Bad Gateway     — Upstream service failure
  503 Service Unavailable — Temporary overload (include Retry-After)
  504 Gateway Timeout — Upstream timeout
```

---

## Authentication Patterns

### JWT with Refresh Tokens

```javascript
// middleware/auth.js
const jwt = require("jsonwebtoken");

const ACCESS_TOKEN_TTL = "15m";
const REFRESH_TOKEN_TTL = "7d";

function generateTokens(user) {
  const accessToken = jwt.sign(
    { userId: user.id, email: user.email, role: user.role },
    process.env.JWT_SECRET,
    { expiresIn: ACCESS_TOKEN_TTL, issuer: "api.example.com" }
  );

  const refreshToken = jwt.sign(
    { userId: user.id, tokenVersion: user.tokenVersion },
    process.env.JWT_REFRESH_SECRET,
    { expiresIn: REFRESH_TOKEN_TTL, issuer: "api.example.com" }
  );

  return { accessToken, refreshToken };
}

function authenticate(req, res, next) {
  const header = req.headers.authorization;
  if (!header?.startsWith("Bearer ")) {
    return ApiResponse.error(res, 401, "UNAUTHORIZED", "Missing access token");
  }

  try {
    const token = header.slice(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET, {
      issuer: "api.example.com",
    });
    req.user = decoded;
    next();
  } catch (err) {
    if (err.name === "TokenExpiredError") {
      return ApiResponse.error(res, 401, "TOKEN_EXPIRED", "Access token expired");
    }
    return ApiResponse.error(res, 401, "UNAUTHORIZED", "Invalid access token");
  }
}

function authorize(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return ApiResponse.error(res, 403, "FORBIDDEN",
        `Requires role: ${roles.join(" or ")}`);
    }
    next();
  };
}

// Routes
router.post("/auth/login", async (req, res) => {
  const { email, password } = req.body;
  const user = await userService.authenticate(email, password);
  if (!user) {
    return ApiResponse.error(res, 401, "INVALID_CREDENTIALS", "Invalid email or password");
  }
  const tokens = generateTokens(user);
  ApiResponse.success(res, { user: sanitize(user), ...tokens });
});

router.post("/auth/refresh", async (req, res) => {
  const { refreshToken } = req.body;
  try {
    const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET);
    const user = await userService.findById(decoded.userId);
    if (!user || user.tokenVersion !== decoded.tokenVersion) {
      return ApiResponse.error(res, 401, "INVALID_TOKEN", "Refresh token revoked");
    }
    const tokens = generateTokens(user);
    ApiResponse.success(res, tokens);
  } catch {
    ApiResponse.error(res, 401, "INVALID_TOKEN", "Invalid refresh token");
  }
});
```

### API Key Authentication

```javascript
// middleware/apiKey.js
function apiKeyAuth(req, res, next) {
  const apiKey = req.headers["x-api-key"] || req.query.api_key;
  if (!apiKey) {
    return ApiResponse.error(res, 401, "MISSING_API_KEY", "API key required");
  }

  // Hash the key for constant-time comparison
  const hashedKey = crypto.createHash("sha256").update(apiKey).digest("hex");
  const client = await db.apiKeys.findOne({ keyHash: hashedKey, active: true });

  if (!client) {
    return ApiResponse.error(res, 401, "INVALID_API_KEY", "Invalid API key");
  }

  req.client = { id: client.clientId, name: client.name, scopes: client.scopes };
  next();
}
```

---

## Rate Limiting

```javascript
// middleware/rateLimiter.js
const Redis = require("ioredis");
const redis = new Redis(process.env.REDIS_URL);

function rateLimiter({ windowMs = 60000, max = 100, keyFn = null } = {}) {
  return async (req, res, next) => {
    const key = keyFn
      ? keyFn(req)
      : `rl:${req.user?.id || req.ip}:${req.route?.path || req.path}`;

    const windowSec = Math.ceil(windowMs / 1000);

    const multi = redis.multi();
    multi.incr(key);
    multi.expire(key, windowSec);
    const [count] = await multi.exec();
    const current = count[1];

    // Set rate limit headers (RFC 6585 / draft-ietf-httpapi-ratelimit-headers)
    res.set({
      "X-RateLimit-Limit": max,
      "X-RateLimit-Remaining": Math.max(0, max - current),
      "X-RateLimit-Reset": new Date(Date.now() + windowMs).toISOString(),
    });

    if (current > max) {
      res.set("Retry-After", windowSec);
      return ApiResponse.error(res, 429, "RATE_LIMITED",
        `Rate limit exceeded. Try again in ${windowSec}s`);
    }

    next();
  };
}

// Different limits for different endpoints
router.use("/api/v1/auth", rateLimiter({ windowMs: 900000, max: 10 }));  // 10/15min
router.use("/api/v1", rateLimiter({ windowMs: 60000, max: 100 }));       // 100/min
router.use("/api/v1/search", rateLimiter({ windowMs: 60000, max: 30 })); // 30/min
```

---

## Request Validation

```javascript
// middleware/validate.js
const Joi = require("joi");

function validate(schema) {
  return (req, res, next) => {
    const targets = { body: req.body, query: req.query, params: req.params };
    const errors = [];

    for (const [target, rules] of Object.entries(schema)) {
      if (!rules) continue;
      const { error, value } = rules.validate(targets[target], {
        abortEarly: false,
        stripUnknown: true,
      });
      if (error) {
        errors.push(
          ...error.details.map((d) => ({
            field: `${target}.${d.path.join(".")}`,
            message: d.message,
            code: "INVALID_FIELD",
          }))
        );
      } else {
        req[target] = value;  // Replace with sanitized values
      }
    }

    if (errors.length > 0) {
      return ApiResponse.error(res, 400, "VALIDATION_ERROR",
        "Request validation failed", errors);
    }
    next();
  };
}

// Schema definitions
const userSchemas = {
  create: {
    body: Joi.object({
      email: Joi.string().email().required(),
      password: Joi.string().min(8)
        .pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/).required()
        .messages({ "string.pattern.base": "Must include uppercase, lowercase, and number" }),
      name: Joi.string().trim().min(1).max(100).required(),
    }),
  },
  list: {
    query: Joi.object({
      page: Joi.number().integer().min(1).default(1),
      limit: Joi.number().integer().min(1).max(100).default(20),
      sort: Joi.string().valid("name", "email", "createdAt", "-name", "-email", "-createdAt")
        .default("createdAt"),
      search: Joi.string().trim().min(1).max(200),
      status: Joi.string().valid("active", "inactive", "suspended"),
    }),
  },
  getById: {
    params: Joi.object({
      id: Joi.string().uuid().required(),
    }),
  },
};

// Usage
router.post("/users", validate(userSchemas.create), userController.create);
router.get("/users", validate(userSchemas.list), userController.list);
router.get("/users/:id", validate(userSchemas.getById), userController.getById);
```

---

## Pagination Patterns

```javascript
// utils/pagination.js

// Offset-based pagination (simple, good for most cases)
async function paginateOffset(model, query, { page = 1, limit = 20 }) {
  const offset = (page - 1) * limit;
  const [data, total] = await Promise.all([
    model.find(query).skip(offset).limit(limit),
    model.countDocuments(query),
  ]);

  return {
    data,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit),
      hasNext: page * limit < total,
      hasPrev: page > 1,
    },
  };
}

// Cursor-based pagination (better for large datasets, real-time feeds)
async function paginateCursor(model, query, { cursor, limit = 20, sortField = "_id" }) {
  const filter = cursor
    ? { ...query, [sortField]: { $gt: cursor } }
    : query;

  const data = await model.find(filter).sort({ [sortField]: 1 }).limit(limit + 1);

  const hasNext = data.length > limit;
  if (hasNext) data.pop();

  return {
    data,
    pagination: {
      limit,
      hasNext,
      nextCursor: hasNext ? data[data.length - 1][sortField] : null,
    },
  };
}
```

---

## OpenAPI Specification

```yaml
# openapi.yaml
openapi: 3.1.0
info:
  title: Example API
  version: 1.0.0
  description: Production REST API
  contact:
    email: api@example.com
  license:
    name: MIT

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

paths:
  /users:
    get:
      operationId: listUsers
      summary: List users
      tags: [Users]
      security:
        - bearerAuth: []
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - name: search
          in: query
          schema:
            type: string
            maxLength: 200
      responses:
        '200':
          description: Users list
          content:
            application/json:
              schema:
                allOf:
                  - $ref: '#/components/schemas/SuccessResponse'
                  - type: object
                    properties:
                      data:
                        type: array
                        items:
                          $ref: '#/components/schemas/User'
        '401':
          $ref: '#/components/responses/Unauthorized'

    post:
      operationId: createUser
      summary: Create a user
      tags: [Users]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          headers:
            Location:
              schema:
                type: string
                example: /api/v1/users/550e8400-e29b-41d4-a716-446655440000
        '400':
          $ref: '#/components/responses/ValidationError'
        '409':
          $ref: '#/components/responses/Conflict'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKey:
      type: apiKey
      in: header
      name: X-API-Key

  parameters:
    PageParam:
      name: page
      in: query
      schema:
        type: integer
        minimum: 1
        default: 1
    LimitParam:
      name: limit
      in: query
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 20

  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
        email:
          type: string
          format: email
        name:
          type: string
        role:
          type: string
          enum: [user, admin, manager]
        status:
          type: string
          enum: [active, inactive, suspended]
        createdAt:
          type: string
          format: date-time

    CreateUserRequest:
      type: object
      required: [email, password, name]
      properties:
        email:
          type: string
          format: email
        password:
          type: string
          minLength: 8
        name:
          type: string
          minLength: 1
          maxLength: 100

    SuccessResponse:
      type: object
      properties:
        status:
          type: string
          enum: [success]
        meta:
          type: object
          properties:
            requestId:
              type: string
            timestamp:
              type: string
              format: date-time

    ErrorResponse:
      type: object
      properties:
        status:
          type: string
          enum: [error]
        error:
          type: object
          properties:
            code:
              type: string
            message:
              type: string
            details:
              type: array
              items:
                type: object
                properties:
                  field:
                    type: string
                  message:
                    type: string

  responses:
    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
    ValidationError:
      description: Request validation failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
    Conflict:
      description: Resource conflict
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
```

---

## API Versioning Strategy

```javascript
// middleware/versioning.js

// URL path versioning (recommended — explicit, cacheable)
app.use("/api/v1", v1Router);
app.use("/api/v2", v2Router);

// Header versioning (useful for non-breaking changes)
function apiVersion(req, res, next) {
  const version = req.headers["api-version"] || "1";
  req.apiVersion = parseInt(version, 10);
  res.set("API-Version", version);
  next();
}

// Deprecation headers
function deprecated(sunset) {
  return (req, res, next) => {
    res.set("Deprecation", "true");
    res.set("Sunset", sunset);  // RFC 8594
    res.set("Link", '</api/v2/users>; rel="successor-version"');
    next();
  };
}

// Apply deprecation to v1
v1Router.use("/users", deprecated("2026-01-01T00:00:00Z"));
```

---

## Global Error Handler

```javascript
// middleware/errorHandler.js

function errorHandler(err, req, res, next) {
  // Log internal details
  const requestId = res.locals.requestId;
  console.error(`[${requestId}] ${err.stack}`);

  // Known operational errors
  if (err.isOperational) {
    return ApiResponse.error(res, err.statusCode, err.code, err.message, err.details);
  }

  // Mongoose validation errors
  if (err.name === "ValidationError") {
    const details = Object.values(err.errors).map((e) => ({
      field: e.path,
      message: e.message,
      code: "INVALID_FIELD",
    }));
    return ApiResponse.error(res, 400, "VALIDATION_ERROR", "Validation failed", details);
  }

  // Duplicate key (MongoDB)
  if (err.code === 11000) {
    const field = Object.keys(err.keyPattern)[0];
    return ApiResponse.error(res, 409, "CONFLICT",
      `A resource with this ${field} already exists`);
  }

  // JWT errors
  if (err.name === "JsonWebTokenError") {
    return ApiResponse.error(res, 401, "UNAUTHORIZED", "Invalid token");
  }

  // Default: don't leak internals
  ApiResponse.error(res, 500, "INTERNAL_ERROR",
    process.env.NODE_ENV === "production"
      ? "An unexpected error occurred"
      : err.message
  );
}

// 404 handler — must be last route
function notFoundHandler(req, res) {
  ApiResponse.error(res, 404, "NOT_FOUND",
    `Route ${req.method} ${req.originalUrl} not found`);
}
```

---

## API Testing

```javascript
// tests/api/users.test.js
const request = require("supertest");
const app = require("../../app");

describe("POST /api/v1/users", () => {
  it("creates a user with valid data", async () => {
    const res = await request(app)
      .post("/api/v1/users")
      .set("Authorization", `Bearer ${adminToken}`)
      .send({ email: "new@example.com", password: "Secure123!", name: "Test User" })
      .expect(201);

    expect(res.body.status).toBe("success");
    expect(res.body.data).toHaveProperty("id");
    expect(res.body.data.email).toBe("new@example.com");
    expect(res.body.data).not.toHaveProperty("password");
    expect(res.headers.location).toMatch(/\/api\/v1\/users\/.+/);
  });

  it("rejects invalid email", async () => {
    const res = await request(app)
      .post("/api/v1/users")
      .set("Authorization", `Bearer ${adminToken}`)
      .send({ email: "not-an-email", password: "Secure123!", name: "Test" })
      .expect(400);

    expect(res.body.status).toBe("error");
    expect(res.body.error.code).toBe("VALIDATION_ERROR");
    expect(res.body.error.details[0].field).toBe("body.email");
  });

  it("returns 409 on duplicate email", async () => {
    await request(app)
      .post("/api/v1/users")
      .set("Authorization", `Bearer ${adminToken}`)
      .send({ email: existingUser.email, password: "Secure123!", name: "Test" })
      .expect(409);
  });

  it("returns 401 without auth", async () => {
    await request(app)
      .post("/api/v1/users")
      .send({ email: "x@x.com", password: "Secure123!", name: "Test" })
      .expect(401);
  });

  it("returns 429 when rate limited", async () => {
    const promises = Array(15).fill().map(() =>
      request(app)
        .post("/api/v1/auth/login")
        .send({ email: "x@x.com", password: "wrong" })
    );
    const responses = await Promise.all(promises);
    const rateLimited = responses.filter((r) => r.status === 429);
    expect(rateLimited.length).toBeGreaterThan(0);
    expect(rateLimited[0].headers["retry-after"]).toBeDefined();
  });
});
```

---

## Anti-Patterns to Avoid

1. ❌ Verbs in URLs — use `POST /orders` not `POST /createOrder`
2. ❌ Inconsistent response shapes — always use the same envelope
3. ❌ Returning 200 for errors — use proper status codes
4. ❌ Exposing internal IDs or stack traces — sanitize all error responses
5. ❌ No pagination on list endpoints — unbounded queries kill databases
6. ❌ Missing rate limiting — one aggressive client can take down your API
7. ❌ Accepting unvalidated input — every field must be validated and sanitized
8. ❌ No request IDs — debugging becomes impossible without correlation
9. ❌ Breaking changes without versioning — always version your API
10. ❌ Documentation that drifts from implementation — generate from code/spec

---

*Adapted from buildwithclaude by Dave Poon (MIT)*
