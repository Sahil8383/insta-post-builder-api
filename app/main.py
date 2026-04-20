"""FastAPI entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import ensure_posts_schema
from app.routers import posts


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_posts_schema()
    yield


app = FastAPI(title="Insta Post Builder API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(posts.router)
