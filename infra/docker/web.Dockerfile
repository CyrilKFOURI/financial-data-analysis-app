# Web app: Vite build served by Caddy, which also proxies /v1 and /mcp to the API (M31 §4).
# Build from the repository root: docker build -f infra/docker/web.Dockerfile .
FROM node:22-alpine AS build
RUN corepack enable && corepack prepare pnpm@12.15.0 --activate
WORKDIR /web
COPY apps/web/package.json apps/web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY apps/web ./
RUN pnpm build

FROM caddy:2.10-alpine
COPY infra/docker/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /web/dist /srv
